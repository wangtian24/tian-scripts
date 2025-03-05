import asyncio
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import aiohttp
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from pydantic import BaseModel
from upstash_redis.asyncio import Redis

from ypl.backend.llm.labeler import LLMLabeler
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.utils.utils import StopWatch
from ypl.db.redis import get_upstash_redis_client

YOUTUBE_VIDEOS_FOR_CHAT_KEY_FORMAT = "youtube_videos_for_chat:{chat_id}"
YOUTUBE_TRANSCRIPT_FOR_VIDEO_ID_KEY_FORMAT = "youtube_transcript_for_video_id:{video_id}"

YOUTUBE_VIDEO_LABELLING_TIMEOUT_SECS = 1.5
YOUTUBE_VIDEO_LABELLING_MODEL = "llama-3.3-70b-versatile"  # This groq model is one of the fastest (200ms p50).
YOUTUBE_ENTRIES_REDIS_TTL_SECS = 7 * 24 * 60 * 60  # 7 days.


class YoutubeProcessingStatus(Enum):
    NONE = "none"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class YoutubeTranscript(BaseModel):
    video_id: str
    status: YoutubeProcessingStatus = YoutubeProcessingStatus.NONE
    transcript: str = ""
    failure_explanation: str = ""

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
        questions about length of the video, timestamp when certain topic is
        mentioned etc.

        The transcript of this video is included below as sequence of segments. Each
        segment is timestamped and has the following format:

        start timestamp in seconds
        Text of the transcript in one or more lines.

        Transcript for the video {video_id}:
        {transcript}
    """
)

SYSTEM_PROMPT_FOR_MISSING_VIDEO_TRANSCRIPT = PromptTemplate.from_template(
    """
        The user prompt likely refers Youtube with video id '{video_id}',
        e.g. https://www.youtube.com/watch?v={video_id}.
        Inform the user that you could not incorporate it because
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
        timeout_secs: float = YOUTUBE_VIDEO_LABELLING_TIMEOUT_SECS,
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
        reply_json_text = str(output.content).replace("```json", "").replace("```", "")

        return YoutubeLabelerResponse.model_validate_json(reply_json_text)


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
                "message": f"Labelling Youtube videos in user messages for {chat_id}",
                "user_prompts": [m.content for m in user_prompts],
            }
        )

        llm = await get_internal_provider_client(YOUTUBE_VIDEO_LABELLING_MODEL, max_tokens=128)
        labeler = YoutubeVideoLabeler(llm)
        label_resp = await labeler.alabel(user_prompts)

        logging.info({"message": f"Youtube labeler response for {chat_id}", "response": label_resp.model_dump()})

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
        logging.error(f"Error while youtube video processing for {chat_id}: {e}", exc_info=True)

    return []


# Module level private methods:


def _system_prompt_with_video_transcript(transcript_status: YoutubeTranscript) -> str:
    """
    Prompt to be appended for each video id. This is used stored in redis.
    """
    video_id = transcript_status.video_id

    if transcript_status.status == YoutubeProcessingStatus.SUCCEEDED:
        return SYSTEM_PROMPT_WITH_VIDEO_TRANSCRIPT.format(video_id=video_id, transcript=transcript_status.transcript)
    else:
        return SYSTEM_PROMPT_FOR_MISSING_VIDEO_TRANSCRIPT.format(
            video_id=video_id, failure_explanation=transcript_status.failure_explanation
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
        # Note: If the process is killed while writing to redis, the status will be stuck in IN_PROGRESS until expiry.
        logging.info({"message": f"Fetched transcript for {video_id} in chat {chat_id} and saved in redis."})
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
                    "message": f"Found video {video_id} for chat {chat_id} in redis",
                    "video_status": stored_status.status.name,
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

    Returns:
        A list of transcripts as timestamped segments.
    """
    params = {
        "engine": "youtube_transcripts",
        "video_id": video_id,
        "api_key": os.getenv("SEARCHAPI_API_KEY"),
        "lang": lang or "en",  # Try common "en" first.
        "transcript_type": "manual",  # Prefer manual
    }

    url = "https://www.searchapi.io/api/v1/search"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:  # type: ignore
            resp = await response.json()

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
            "message": "Fetched transcript for video",
            "video_id": video_id,
            "lang": resp["search_parameters"]["lang"],
            "searchapi_json_url": resp["search_metadata"]["json_url"],
        }
    )

    return resp["transcripts"]  # type: ignore


async def _fetch_youtube_transcript(video_id: str) -> TranscriptFetcherResponse:
    """
    Fetches transcript for the video id using searchapi.com.
    Returns `TranscriptFetcherResponse` with list of timestamped segments of the transcript.
    This does not raise an exception. Any runtime errors are logged and`is_successful` is set to false.
    """

    is_successful = False
    failure_explanation = ""
    timestamped_transcript = ""

    try:
        stop_watch = StopWatch()

        transcript = await _fetch_transcript_from_searchapi(video_id)

        num_segments = len(transcript)
        max_time_secs = (
            transcript[num_segments - 1]["start"] + transcript[num_segments - 1]["duration"] if num_segments else 0
        )

        # Combine the all segments into single string in the format
        #   start_time_secs
        #   text
        timestamped_transcript = "\n".join([f"{s['start']}\n{s['text']}" for s in transcript])
        is_successful = True

        stop_watch.end("fetch_transcript")
        logging.info(
            {
                "message": f"Fetched video {video_id} in {stop_watch.get_total_time()}ms. ",
                "size": len(timestamped_transcript),
                "video_duration": max_time_secs,
                "num_segments": num_segments,
            }
        )

    except YoutubeTranscriptNotFound:
        failure_explanation = "transcript is not available for the video"
    except Exception as e:
        logging.error(f"Failed to fetch transcript due to {type(e)}: {e}", exc_info=True)
        failure_explanation = "fetch failed with an error"

    return TranscriptFetcherResponse(
        is_successful=is_successful,
        failure_explanation=failure_explanation,
        timestamped_transcript=timestamped_transcript,
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
