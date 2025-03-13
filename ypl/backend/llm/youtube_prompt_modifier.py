import asyncio
import logging
import random
import re
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from pydantic import BaseModel
from upstash_redis.asyncio import Redis

from ypl.backend.llm.labeler import LLMLabeler, MultiLLMLabeler
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.llm.searchapi import fetch_search_api_response
from ypl.backend.utils.utils import StopWatch
from ypl.db.redis import get_upstash_redis_client
from ypl.utils import extract_json_dict_from_text, maybe_truncate

YOUTUBE_VIDEOS_FOR_CHAT_KEY_FORMAT = "youtube_videos_for_chat:{chat_id}"
YOUTUBE_TRANSCRIPT_FOR_VIDEO_ID_KEY_FORMAT = "youtube_transcript_for_video_id:{video_id}"

YOUTUBE_VIDEO_LABELING_TIMEOUT_SECS = 1.5
YOUTUBE_VIDEO_LABELING_MODEL_1 = "llama-3.3-70b-versatile"  # This groq model is one of the fastest (200ms p50).
YOUTUBE_VIDEO_LABELING_MODEL_2 = "gpt-4o-mini"  # A more reliable model, likely slower. The first response is used.
YOUTUBE_ENTRIES_REDIS_TTL_SECS = 7 * 24 * 60 * 60  # 7 days.

_YOUTUBE_MULTI_LABELER = None


async def _get_youtube_multi_labeler() -> MultiLLMLabeler:
    global _YOUTUBE_MULTI_LABELER

    if _YOUTUBE_MULTI_LABELER is None:
        _YOUTUBE_MULTI_LABELER = MultiLLMLabeler(
            labelers={
                YOUTUBE_VIDEO_LABELING_MODEL_1: YoutubeVideoLabeler(
                    await get_internal_provider_client(YOUTUBE_VIDEO_LABELING_MODEL_1, max_tokens=128)
                ),
                YOUTUBE_VIDEO_LABELING_MODEL_2: YoutubeVideoLabeler(
                    await get_internal_provider_client(YOUTUBE_VIDEO_LABELING_MODEL_2, max_tokens=128)
                ),
            },
            timeout_secs=YOUTUBE_VIDEO_LABELING_TIMEOUT_SECS,
            early_terminate_on=[YOUTUBE_VIDEO_LABELING_MODEL_1, YOUTUBE_VIDEO_LABELING_MODEL_2],
        )

    return _YOUTUBE_MULTI_LABELER


class YoutubeProcessingStatus(Enum):
    NONE = "none"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class YoutubeVideoInfo(BaseModel):
    title: str | None = None
    length_seconds: int | None = None
    author: str | None = None
    published_time: str | None = None
    description: str | None = None


class YoutubeTranscript(BaseModel):
    video_id: str
    status: YoutubeProcessingStatus = YoutubeProcessingStatus.NONE
    transcript: str = ""
    failure_explanation: str = ""
    video_info: YoutubeVideoInfo | None = None

    def redis_key(self) -> str:
        return YOUTUBE_TRANSCRIPT_FOR_VIDEO_ID_KEY_FORMAT.format(video_id=self.video_id)


class YoutubeLabelerResponse(BaseModel):
    requires_transcript: bool
    video_ids: list[str]


@dataclass
class TranscriptFetcherResponse:
    is_successful: bool
    failure_explanation: str
    timestamped_transcript: str
    video_info: YoutubeVideoInfo | None = None


YOUTUBE_VIDEO_ID_LABELER_SYSTEM_PROMPT = """
  You are a helpful assistant that determines whether to fetch YouTube video transcripts to
  answer user prompts effectively. Your responses should be in JSON format.

  Consider the following factors:
   - The text transcript of these videos is relevant to reply to the prompt
   - Extract the video ids from the URLs.
   - There can be more than one youtube videos. List all the relevant ones.
   - The text of the transcript has timestamps. This is useful for answering
     questions about length of the video, timestamp of certain topics mentioned etc.

  The response should be in the following JSON format. Do not include any other information.

  {
    "requires_transcript": true/false,
    "video_ids": ["vid_id_1", "vid_id_2"]
  }

  Examples:

  1. User prompt: "What is the summary of https://youtu.be/cVsQLlk-T0s?si=iaDSvyXWjjsYIBSX"
     Response:
         {
          "requires_transcript": true,
          "video_ids": ["cVsQLlk-T0s"]
         }

  2. User prompt: "Where in the video does the presenter talk about the period
       when first mammals appeared: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    Response:
        {
          "requires_transcript": true,
          "video_ids": ["dQw4w9WgXcQ"]
        }
  3. User prompt: "Explain core principles of Thermodynamics"
     Response:
        {
          "requires_transcript": false,
          "video_ids": []
        }

  4. User prompt: "What is the Youtube video id of https://youtu.be/cVsQLlk-T0s?si=iaDSvyXWjjsYIBSX"
     Response:
        {
          "requires_transcript": false,
          "video_ids": []
        }

  5. User prompt: "What colors dominate the background of this video? https://www.youtube.com/watch?v=qwer4321."
     Response:
        {
          "requires_transcript": false,
          "video_ids": []
        }
  6. User prompt: "Compare and contrast important points made in these two videos:
    https://www.youtube.com/watch?v=gFQmapcvgls, https://youtu.be/VuqHl9SDm0s?si=N_AoqhXp4uQM9kQ2"
    Response:
        {
          "requires_transcript": true,
          "video_ids": ["gFQmapcvgls", "N_AoqhXp4uQM9kQ2"]
        }
  7. User prompt: "How long is this video https://youtu.be/VuqHl9SDm0s?si=N_AoqhXp4uQM9kQ2"
    Response:
       {
          "requires_transcript": true,
          "video_ids": ["N_AoqhXp4uQM9kQ2"]
       }
"""

SYSTEM_PROMPT_WITH_VIDEO_TRANSCRIPT = PromptTemplate.from_template(
    """
        The user likely refers Youtube with video id '{video_id}',
        e.g. https://www.youtube.com/watch?v={video_id}.
        The text of the transcript has timestamps. This is useful for answering
        questions about length of the video, timestamp when certain topic is mentioned etc.
        If the user prompt only includes the video and does not say anything else, summarize it for them.

        The following JSON optionally contains information about the video like title, description etc.
        {video_info_json}

        The transcript of this video is included below as sequence of segments. Each
        segment is timestamped and has the following format:

        start timestamp in seconds
        Text of the transcript in one or more lines.

        Transcript for the video {video_id}:

        -- Start of transcript --
        {transcript}
        -- End of transcript --
    """
)

SYSTEM_PROMPT_FOR_MISSING_VIDEO_TRANSCRIPT = PromptTemplate.from_template(
    """
        The user prompt likely refers Youtube with video id '{video_id}',
        e.g. https://www.youtube.com/watch?v={video_id}.

        The following JSON optionally contains information about the video like title, description etc.
        {video_info_json}

        Clearly inform the user that you could not access its transcript because
        {failure_explanation}.
    """
)


class YoutubeVideoLabeler(LLMLabeler[list[HumanMessage], YoutubeLabelerResponse]):
    """
    Labels the user prompt to determine if the transcript of a youtube video is needed.
    It returns the list of video ids relevant to the address the prompt.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        timeout_secs: float = YOUTUBE_VIDEO_LABELING_TIMEOUT_SECS,
    ):
        super().__init__(llm, timeout_secs=timeout_secs, on_error="raise")

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=YOUTUBE_VIDEO_ID_LABELER_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="prompts"),
            ]
        )

        return template | llm  # type: ignore

    def _prepare_input(self, user_prompts: list[HumanMessage]) -> dict[str, Any]:
        return {"prompts": user_prompts}

    def _parse_output(self, output: BaseMessage) -> YoutubeLabelerResponse:
        reply_json_text = extract_json_dict_from_text(str(output.content))

        try:
            return YoutubeLabelerResponse.model_validate_json(reply_json_text)
        except Exception:
            logging.warning(
                {
                    "message": "Failed to parse response from Youtube labeler. Returning empty response.",
                    "labeler_response": maybe_truncate(reply_json_text, 500),
                },
                exc_info=True,
            )
            return YoutubeLabelerResponse(requires_transcript=False, video_ids=[])


class YoutubeTranscriptNotFound(Exception):
    pass


async def maybe_youtube_transcript_messages(chat_id: str, chat_history: list[BaseMessage]) -> list[BaseMessage]:
    """
    If there are any youtube links in the user prompts:
      - Run through a labeler to determine relevant youtube videos.
      - Fetch the transcripts for each of the video ids.
      - The transcript is saved in redis.
      - This is called from each chat completion request, i.e. multiple times for each turn.
         - Only one of these calls will fetch transcript for a video. Other will read from redis.

    Returns a list of system messages to be added to LLM request.
    """

    if not _fast_check_for_youtube_links(chat_history):
        logging.info({"message": f"No Youtube links in chat {chat_id}. Skipping further youtube processing."})
        return []  # Common case. Most chats don't reference youtube videos.

    user_prompts = [m for m in chat_history if isinstance(m, HumanMessage)]

    try:
        logging.info(
            {
                "message": f"Labeling Youtube videos in user messages for {chat_id}",
                "user_prompts": [maybe_truncate(str(m.content), 500) for m in user_prompts],
            }
        )

        multi_labeler = await _get_youtube_multi_labeler()

        results: dict[str, Any] = await multi_labeler.alabel(user_prompts)
        successful_results = [(m, r) for m, r in results.items() if isinstance(r, YoutubeLabelerResponse)]
        if not successful_results:  # Both failed. Raise one of them.
            raise random.choice(list(results.values()))

        model, label_resp = random.choice(successful_results)  # Pick one at random. Usually there is only one.

        logging.info({"message": f"Youtube videos returned for {chat_id} by {model}: {label_resp.video_ids}"})

        # Note: We are invoking labeler for each chat_completion. We only need to call to once per turn for all the
        # selected models. We could use redis to avoid extra labeler invocations to void it. Since very small fraction
        # of chats have this overhead, it is ok.

        if label_resp.video_ids:
            transcript_results = await asyncio.gather(
                *[_process_transcript_for_video_id(chat_id, video_id) for video_id in label_resp.video_ids]
            )

            messages: list[BaseMessage] = [
                SystemMessage(content=_system_prompt_with_video_transcript(transcript))
                for transcript in transcript_results
            ]
            logging.info(
                {
                    "message": f"Adding Youtube system prompt(s) for {chat_id}",
                    "videos": [
                        {"id": v, "size": len(m.content)} for v, m in zip(label_resp.video_ids, messages, strict=True)
                    ],
                }
            )

            return messages

    except Exception as e:
        logging.error({"message": f"Error while processing youtube video for {chat_id}: {e}"}, exc_info=True)

    return []


# Module level private methods:


def _system_prompt_with_video_transcript(transcript_status: YoutubeTranscript) -> str:
    """
    Prompt to be appended for each video id. This is used stored in redis.
    """
    video_id = transcript_status.video_id
    video_info_json = (
        transcript_status.video_info.model_dump_json(exclude_none=True)
        if transcript_status.video_info
        else """{"message": "No video info available"}"""
    )

    if transcript_status.status == YoutubeProcessingStatus.SUCCEEDED:
        return SYSTEM_PROMPT_WITH_VIDEO_TRANSCRIPT.format(
            video_id=video_id,
            transcript=transcript_status.transcript,
            video_info_json=video_info_json,
        )
    else:
        return SYSTEM_PROMPT_FOR_MISSING_VIDEO_TRANSCRIPT.format(
            video_id=video_id,
            failure_explanation=transcript_status.failure_explanation,
            video_info_json=video_info_json,
        )


# Look for 'youtube.com/' or 'youtu.be/' in the prompt. Case insensitive.
# This is intentionally simple without requiring https:// or www. etc.
_YOUTUBE_LINK_REGEX = re.compile(r"(?i)(?:youtube\.com/|youtu\.be/)")


def _fast_check_for_youtube_links(chat_history: list[BaseMessage]) -> bool:
    """
    Checks if there are any youtube links mentioned in the chat history.
    """
    return any(
        re.search(_YOUTUBE_LINK_REGEX, m.content)
        for m in chat_history
        if isinstance(m, HumanMessage) and isinstance(m.content, str)
    )


async def _process_transcript_for_video_id(chat_id: str, video_id: str) -> YoutubeTranscript:
    """
    Fetches the transcript for a video id and writes it to redis.
    """
    redis_client = await get_upstash_redis_client()

    transcript_status = YoutubeTranscript(
        video_id=video_id,
        status=YoutubeProcessingStatus.IN_PROGRESS,
    )

    is_set = await redis_client.set(
        transcript_status.redis_key(), transcript_status.model_dump_json(), nx=True, ex=YOUTUBE_ENTRIES_REDIS_TTL_SECS
    )

    if is_set:
        # Fetch the transcript here.
        logging.info({"message": f"Fetching transcript for video {video_id} in chat {chat_id}"})
        fetcher_response = await _fetch_youtube_transcript(video_id)
        transcript_status.video_info = fetcher_response.video_info

        if fetcher_response.is_successful:
            transcript_status.status = YoutubeProcessingStatus.SUCCEEDED
            transcript_status.transcript = fetcher_response.timestamped_transcript
        else:
            transcript_status.status = YoutubeProcessingStatus.FAILED
            transcript_status.failure_explanation = fetcher_response.failure_explanation
        # Update redis
        await redis_client.set(
            transcript_status.redis_key(), transcript_status.model_dump_json(), ex=YOUTUBE_ENTRIES_REDIS_TTL_SECS
        )
        # Note: If the process is killed before writing to redis, the status will be stuck in IN_PROGRESS until expiry.
        logging.info(
            {
                "message": (
                    f"Fetched transcript for {video_id} for chat {chat_id} "
                    f"with status {transcript_status.status.name} and saved in redis."
                ),
                "video_title": transcript_status.video_info.title if transcript_status.video_info else None,
            }
        )
        return transcript_status

    else:
        # read the existing value (or wait if it is being processed)
        logging.info({"message": f"Looking for video {video_id} for chat {chat_id} in redis."})
        stored_status = await _fetch_redis_value_with_retry(
            redis_client,
            YoutubeTranscript(video_id=video_id).redis_key(),
            deserializer=YoutubeTranscript.model_validate_json,
            should_return=lambda v: v.status != YoutubeProcessingStatus.IN_PROGRESS,
            timeout=6.0,  # Give a longer timeout for transcript to be fetched.
        )

        if stored_status is None:
            # Likely ongoing fetch took too long.
            logging.info(
                {"message": f"Could not find video {video_id} in redis for {chat_id}. Likely fetch is taking too long."}
            )
            return YoutubeTranscript(
                video_id=video_id,
                status=YoutubeProcessingStatus.FAILED,
                failure_explanation="fetch failed or timed out",
            )
        else:
            logging.info(
                {
                    "message": f"Found video {video_id} for {chat_id} in redis with status {stored_status.status.name}",
                    "video_status": stored_status.status.name,
                    "video_title": stored_status.video_info.title if stored_status.video_info else None,
                    "transcript_size": len(stored_status.transcript),
                    "failure_explanation": stored_status.failure_explanation,
                }
            )
            return stored_status


_LANGUAGE_NOT_TRANSCRIBED_ERROR_STRING = "Selected language hasn't been transcribed. Check `available_languages`"
_NO_TRANSLATIONS_AVAILABLE_ERROR_STRING = "there are no translations available"


async def _fetch_transcript_from_searchapi(video_id: str, lang: str | None = None) -> dict:
    """
    Fetches transcript from searchapi.com. When lang is not provided, it tries 'en' first.
    If `en` is not available, the reply from searchapi will have a list of available languages.
    It then tries again with another English entry if found. If there is no English entry, it picks
    the first entry from the list of available languages for the second attempt.
    E.g. some videos have English subtitles but the language code is "en-US", rather than just "en".

    See https://www.searchapi.io/docs/youtube-transcripts for more API details and sample responses.

    Returns:
        A list of transcripts as timestamped segments.
    """
    params = {
        "engine": "youtube_transcripts",
        "video_id": video_id,
        "lang": lang or "en",  # Try common "en" first.
        "transcript_type": "manual",  # Prefer manual
    }

    resp = await fetch_search_api_response(params)

    if "error" in resp:
        if _LANGUAGE_NOT_TRANSCRIBED_ERROR_STRING in resp["error"] and lang is None:
            # Try again picking the next best match for language.
            # Sometimes language is English, but code is "en-US", rather than "en".
            # If no english transcript is found, pick the first one.
            available_languages: list[dict] = resp["available_languages"]

            en_like = [entry for entry in available_languages if entry["name"].startswith("English")]

            best_match = en_like[0] if en_like else available_languages[0]

            logging.info(f"Did not find 'en' transcript for {video_id}. Trying with next best match {best_match}.")

            return await _fetch_transcript_from_searchapi(video_id, best_match["lang"])

        elif _NO_TRANSLATIONS_AVAILABLE_ERROR_STRING in resp["error"]:
            raise YoutubeTranscriptNotFound(f"No transcripts found for {video_id}")

        else:
            raise RuntimeError(resp["error"])

    logging.info(
        {
            "message": f"Successfully fetched transcript for video {video_id}",
            "video_id": video_id,
            "lang": resp["search_parameters"]["lang"],
            "searchapi_json_url": resp["search_metadata"]["json_url"],
        }
    )

    return resp["transcripts"]  # type: ignore


async def _fetch_youtube_video_info_from_searchapi(video_id: str) -> YoutubeVideoInfo | None:
    """
    Fetches video info from searchapi.com. If there is an error reported by searchapi, returns None.
    See https://www.searchapi.io/docs/youtube-video for more API details and sample responses.
    """
    params = {
        "engine": "youtube_video",
        "video_id": video_id,
    }

    resp = await fetch_search_api_response(params)

    logging.info(
        {
            "message": f"Fetched Youtube video info for {video_id}",
            "title": resp.get("video", {}).get("title"),
            "searchapi_json_url": resp["search_metadata"]["json_url"],
            "error": resp.get("error", "No error"),
        }
    )

    if "error" in resp:
        logging.warning({"message": f"Error while fetching video info for {video_id}: {resp['error']}"})
        return None
    elif "video" in resp:
        video = resp["video"]
        return YoutubeVideoInfo(
            title=video.get("title"),
            length_seconds=video.get("length_seconds"),
            author=video.get("author"),
            published_time=video.get("published_time"),
            description=maybe_truncate(video.get("description"), 1000),
        )
    else:
        raise RuntimeError(f"Unexpected response from searchapi.com for video {video_id}: {resp}")


async def _fetch_youtube_transcript(video_id: str) -> TranscriptFetcherResponse:
    """
    Fetches transcript for the video id using searchapi.com.
    Returns `TranscriptFetcherResponse` with list of timestamped segments of the transcript.
    This does not raise an exception. Any runtime errors are logged and`is_successful` is set to false.
    """

    is_successful = False
    failure_explanation = ""
    timestamped_transcript = ""

    stop_watch = StopWatch()

    transcript, video_info = await asyncio.gather(
        _fetch_transcript_from_searchapi(video_id),
        _fetch_youtube_video_info_from_searchapi(video_id),
        return_exceptions=True,
    )

    try:
        # Process transcript first
        if isinstance(transcript, Exception):
            raise transcript

        num_segments = len(transcript)
        max_time_secs = (
            transcript[num_segments - 1]["start"] + transcript[num_segments - 1]["duration"] if num_segments else 0
        )

        # Combine the all segments into single string in the format
        #   start_time_in_seconds
        #   text
        timestamped_transcript = "\n" + "\n\n".join([f"{round(s['start'])}\n{s['text']}" for s in transcript])
        is_successful = True

        stop_watch.end("fetch_transcript")
        logging.info(
            {
                "message": f"Fetched video transcript for {video_id} in {stop_watch.get_total_time()}ms. ",
                "size": len(timestamped_transcript),
                "video_duration": max_time_secs,
                "num_segments": num_segments,
            }
        )

    except YoutubeTranscriptNotFound:
        logging.warning({"message": f"No transcripts are available for video {video_id}"})
        failure_explanation = "transcript is not available for the video"
    except Exception as e:
        logging.error({"message": f"Failed to fetch transcript for {video_id} with exception {e}"}, exc_info=True)
        failure_explanation = "fetch failed with an error"

    if isinstance(video_info, Exception):
        logging.warning(
            {
                "message": f"exception while fetching video info for {video_id}",
                "exception": "".join(traceback.format_exception(video_info)),
            }
        )
        video_info = None

    return TranscriptFetcherResponse(
        is_successful=is_successful,
        failure_explanation=failure_explanation,
        timestamped_transcript=timestamped_transcript,
        video_info=video_info,
    )


T = TypeVar("T")


async def _fetch_redis_value_with_retry(
    redis_client: Redis,
    key: str,
    deserializer: Callable[[str], T],
    should_return: Callable[[T], bool],
    timeout: float = 2.0,
    retry_delay: float = 0.1,
) -> T | None:
    """
    Fetch a value from Redis, deserialize it, and apply conditional retry logic.
    Returns:
        The deserialized value if found and predicate is satisfied, or None otherwise
    """
    start_time = asyncio.get_event_loop().time()

    while asyncio.get_event_loop().time() - start_time < timeout:
        raw_value = await redis_client.get(key)

        if raw_value is None:  # If value not found, return None immediately. We could to wait a bit as well.
            return None

        value = deserializer(raw_value)
        if should_return(value):
            return value

        await asyncio.sleep(retry_delay)

    return None
