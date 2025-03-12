import logging
import re
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import aiohttp
from langchain.callbacks.base import AsyncCallbackHandler
from sqlalchemy import select, update
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ypl.backend.attachments.upload import upload_original
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.utils.async_utils import create_background_task
from ypl.backend.utils.json import json_dumps
from ypl.db.attachments import Attachment
from ypl.db.chats import ChatMessage, MessageType

"""
Handle generated images from LLM providers which we get as a URL. We download the image, upload it to GCS,
and store the url to the attachments table in DB. This entry will later be used to replace the url in the message so
we are able to view the image when we load the chat in the future.
"""


async def _download_url(file_url: str) -> tuple[bytes, str] | None:
    """
    Downloads a file from a URL and returns the content and the content type.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                response_content = await response.read()
                response_status = response.status
                response_headers = response.headers

                if response_status == 200:
                    content_type = response_headers.get("Content-Type")
                    if content_type is not None and content_type.startswith("image/"):
                        return response_content, content_type
                    else:
                        logging.warning(f"Empty or non-image content type while downloading from {file_url}")
                        return None
                else:
                    logging.warning(f"Error downloading image from {file_url}: {response_status}")
                    return None
    except Exception as e:
        logging.warning(f"Error downloading image from {file_url}: {e}")
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.2),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type(Exception),
)
async def _upload_to_gcs(content: bytes, content_type: str, file_url: str) -> str | None:
    """
    Downloads a file from an external URL and uploads it to GCS, return url of the uploaded file.
    Returns None if any error occurs.
    """
    gcs_file_uuid = uuid4()
    gcs_bucket_path = settings.ATTACHMENT_BUCKET_GEN_IMAGES
    gcs_bucket_name, *gcs_path_parts = gcs_bucket_path.replace("gs://", "").split("/")

    await upload_original(
        gcs_file_uuid,
        gcs_bucket_name,
        gcs_path_parts,
        content,
        content_type,
        # this name will expire soon but we are just keeping it here for reference
        file_url,
    )
    return f"gs://{gcs_bucket_name}/{'/'.join(gcs_path_parts)}/{gcs_file_uuid}"


async def _write_to_attachment_db(message_id: UUID, gcs_url: str, original_file_url: str, content_type: str) -> UUID:
    async with get_async_session() as session:
        attachment = Attachment(
            chat_message_id=message_id,
            url=gcs_url,
            file_name=original_file_url,
            content_type=content_type,
        )
        session.add(attachment)
        await session.commit()
        await session.refresh(attachment)
        return attachment.attachment_id


async def persist_generated_image(file_url: str, message_id: UUID) -> UUID | None:
    # download from API's provided URL
    bytes_and_content_type = await _download_url(file_url)
    if not bytes_and_content_type:
        return None

    # upload to GCS and get an internal url
    content, content_type = bytes_and_content_type
    gcs_url = await _upload_to_gcs(content, content_type, file_url)
    if gcs_url is None:
        return None

    # store the new url information to DB, return attachment id
    attachment_id = await _write_to_attachment_db(message_id, gcs_url, file_url, content_type)

    log_dict = {
        "message": f"Persisted generated image for message {message_id}",
        "message_id": message_id,
        "file_url_from_api": file_url,
        "persisted_gcs_url": gcs_url,
        "attachment_id": attachment_id,
        "content_type": content_type,
    }
    logging.info(json_dumps(log_dict))

    return attachment_id


URL_CHECK_INTERVAL = timedelta(minutes=65)


class ImageGenCallback(AsyncCallbackHandler):
    def __init__(self, message_id: UUID):
        self.message_id = message_id

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        create_background_task(persist_generated_image(token, self.message_id))


async def do_backfill_gen_image_urls() -> None:
    """
    Update gen_image_url in message table, these messages are from image-generating models and they have an ephemeral
    url that needs to be replaced with a persistent url on GCS.
    """
    async with get_async_session() as session:
        query = (
            select(ChatMessage.message_id, ChatMessage.content, Attachment.url)  # type: ignore
            .join(Attachment, ChatMessage.message_id == Attachment.chat_message_id)
            .where(
                ChatMessage.deleted_at.is_(None),  # type: ignore
                ChatMessage.message_type == MessageType.ASSISTANT_MESSAGE,
                Attachment.url.isnot(None),  # type: ignore
                Attachment.deleted_at.is_(None),  # type: ignore
                Attachment.created_at > datetime.now() - URL_CHECK_INTERVAL,  # type: ignore
            )
        )
        results = (await session.exec(query)).all()
        """
        Replacing all urls
        The chat message content looks like:

            <yapp class="image">
                {
                    "url": "https://.....",
                    "caption": "Generated image"
                }
            </yapp>

        We will replace the url part with a new gs:// resource url.
        """
        for result in results:
            message_id, content, attachment_url = result
            new_content = re.sub(r'("url"\s*:\s*)"[^"]*"', f'\\1"{attachment_url}"', content)
            await session.exec(
                update(ChatMessage).where(ChatMessage.message_id == message_id).values(content=new_content)
            )  # type: ignore
            print(f"Updated message {message_id}")

        await session.commit()

        logging.info(f"Backfilled {len(results)} messages with new persisted image urls.")
