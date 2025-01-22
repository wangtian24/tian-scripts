import asyncio
import base64
import logging
from datetime import datetime
from typing import Any, TypedDict

from gcloud.aio.storage import Storage
from langchain_core.messages import BaseMessage, HumanMessage

from ypl.backend.llm.provider.provider_clients import get_model_provider_tuple
from ypl.backend.prompts import IMAGE_POLYFILL_PROMPT
from ypl.backend.utils.json import json_dumps
from ypl.db.attachments import Attachment


class TransformOptions(TypedDict, total=False):
    use_signed_url: bool
    use_thumbnails: bool


DEFAULT_OPTIONS: TransformOptions = {
    "use_signed_url": False,
    "use_thumbnails": False,
}


async def download_attachment(attachment: Attachment, use_thumbnails: bool = False) -> bytes:
    start_time = datetime.now()
    try:
        gcs_path = attachment.url if not use_thumbnails else attachment.thumbnail_url

        if not gcs_path or not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_path}")

        bucket_name, *path_parts = gcs_path.replace("gs://", "").split("/")
        if not bucket_name or not path_parts:
            raise ValueError(f"Invalid GCS path format: {gcs_path}")

        try:
            logging.info(f"Attachments: Downloading attachment {attachment.attachment_id} from {gcs_path}")
            async with Storage() as async_client:
                blob = await async_client.download(
                    bucket=bucket_name,
                    object_name=f"{'/'.join(path_parts)}",
                )
                return blob

        except Exception as e:
            log_dict = {
                "message": "GCS download failed",
                "error": str(e),
                "attachment_id": str(attachment.attachment_id),
                "gcs_path": gcs_path,
                "file_name": attachment.file_name,
                "content_type": attachment.content_type,
            }
            logging.exception(json_dumps(log_dict))
            raise RuntimeError(f"Failed to download file from GCS: {str(e)}") from e
        finally:
            end_time = datetime.now()
            logging.info(f"Attachments: {attachment.attachment_id} - GCS download took {end_time - start_time} seconds")

    except ValueError as e:
        log_dict = {
            "message": "Invalid GCS path",
            "error": str(e),
            "attachment_id": str(attachment.attachment_id),
            "url": attachment.url,
            "file_name": attachment.file_name,
            "content_type": attachment.content_type,
        }
        logging.error(json_dumps(log_dict))
        raise


async def generate_image_part(
    attachment: Attachment, use_signed_url: bool = False, use_thumbnails: bool = False
) -> dict[str, Any]:
    if use_signed_url:
        url = await get_image_signed_url(attachment)
    else:
        image_bytes = await download_attachment(attachment, use_thumbnails)
        mime_type = "image/png" if use_thumbnails else attachment.content_type
        url = f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": url,
        },
    }


async def transform_user_messages(
    messages: list[BaseMessage],
    model_name: str,
    options: TransformOptions = DEFAULT_OPTIONS,
) -> list[BaseMessage]:
    model_provider = get_model_provider_tuple(model_name)
    if not model_provider:
        raise ValueError(f"No model-provider configuration found for: {model_name}")

    model = model_provider[0]

    all_attachments = [
        attachment for message in messages for attachment in message.additional_kwargs.get("attachments", [])
    ]
    chat_has_attachments = len(all_attachments) > 0

    supports_images = model.supports_images()

    if not chat_has_attachments:
        return [m if not isinstance(m, HumanMessage) else HumanMessage(content=m.content) for m in messages]

    if not supports_images:
        logging.warning(
            "Attachments: Model does not support images. Skipping transformation."
            + f"Model: {model.name} - Provider: {model_provider[1]}"
        )
        return [m if not isinstance(m, HumanMessage) else HumanMessage(content=m.content) for m in messages]

    transformed_messages: list[BaseMessage] = []

    use_signed_url = options.get("use_signed_url", DEFAULT_OPTIONS["use_signed_url"])
    use_thumbnails = options.get("use_thumbnails", DEFAULT_OPTIONS["use_thumbnails"])

    attachment_id_to_content_dict: dict[str, dict[str, Any]] = {}
    attachment_tasks = [
        generate_image_part(attachment, use_signed_url, use_thumbnails) for attachment in all_attachments
    ]
    results = await asyncio.gather(*attachment_tasks, return_exceptions=True)
    for attachment, result in zip(all_attachments, results, strict=True):
        if isinstance(result, BaseException):
            logging.exception(f"Attachments: skipping attachment: {attachment.attachment_id} - {str(result)}")
            continue
        attachment_id_to_content_dict[attachment.attachment_id] = result

    for message in messages:
        if not isinstance(message, HumanMessage):
            transformed_messages.append(message)
            continue

        attachments = message.additional_kwargs.get("attachments", [])
        if len(attachments) == 0:
            transformed_messages.append(HumanMessage(content=message.content))
            continue

        content: list[dict[str, Any]] = []

        for attachment in attachments:
            image_content = attachment_id_to_content_dict.get(attachment.attachment_id)
            if image_content:
                content.append(image_content)
        content.append({"type": "text", "text": str(message.content)})

        transformed_messages.append(HumanMessage(content=content))  # type: ignore
    return transformed_messages


def get_image_polyfill_prompt(attachment: Attachment) -> str:
    if not attachment.attachment_metadata:
        return ""
    prompt = f"""
    file_name: {attachment.file_name}
    description: {attachment.attachment_metadata.get("description", "")}
    """
    return prompt.strip()


def get_images_polyfill_prompt(attachments: list[Attachment], question: str) -> str:
    if not attachments:
        return question
    image_metadata_prompt = "\n---\n".join(
        [get_image_polyfill_prompt(attachment) for attachment in attachments if attachment.attachment_metadata]
    ).strip()
    if not image_metadata_prompt:
        return question
    return IMAGE_POLYFILL_PROMPT.format(image_metadata_prompt=image_metadata_prompt, question=question)


async def get_image_signed_url(attachment: Attachment) -> str:
    start_time = datetime.now()
    try:
        gcs_path = attachment.url
        if not gcs_path or not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_path}")

        bucket_name, *path_parts = gcs_path.replace("gs://", "").split("/")
        if not bucket_name or not path_parts:
            raise ValueError(f"Invalid GCS path format: {gcs_path}")

        try:
            async with Storage() as async_client:
                bucket = async_client.get_bucket(bucket_name)
                blob = await bucket.get_blob(f"{'/'.join(path_parts)}")
                url = await blob.get_signed_url(expiration=604800)
                return url

        except Exception as e:
            log_dict = {
                "message": "GCS signed url failed",
                "error": str(e),
                "attachment_id": str(attachment.attachment_id),
                "gcs_path": gcs_path,
                "file_name": attachment.file_name,
                "content_type": attachment.content_type,
            }
            logging.exception(json_dumps(log_dict))
            raise RuntimeError(f"Failed to download file from GCS: {str(e)}") from e
        finally:
            end_time = datetime.now()
            logging.info(f"Attachments: {attachment.attachment_id} - GCS download took {end_time - start_time} seconds")

    except ValueError as e:
        log_dict = {
            "message": "Invalid GCS path",
            "error": str(e),
            "attachment_id": str(attachment.attachment_id),
            "url": attachment.url,
            "file_name": attachment.file_name,
            "content_type": attachment.content_type,
        }
        logging.error(json_dumps(log_dict))
        raise
