import asyncio
import base64
import logging
from datetime import datetime
from typing import Any

from gcloud.aio.storage import Storage
from langchain_core.messages import BaseMessage, HumanMessage

from ypl.backend.llm.provider.provider_clients import get_model_provider_tuple
from ypl.backend.prompts import IMAGE_POLYFILL_PROMPT
from ypl.backend.utils.json import json_dumps
from ypl.db.attachments import Attachment


async def download_attachment(attachment: Attachment) -> bytes:
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


async def generic_image_transformer(attachment: Attachment) -> dict[str, Any]:
    image_bytes = await download_attachment(attachment)
    url = f"data:{attachment.content_type};base64," + base64.b64encode(image_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": url,
        },
    }


async def transform_user_mesages(messages: list[BaseMessage], model_name: str) -> list[BaseMessage]:
    model_provider = get_model_provider_tuple(model_name)
    if not model_provider:
        raise ValueError(f"No model-provider configuration found for: {model_name}")

    model = model_provider[0]

    chat_has_attachments = any(message.additional_kwargs.get("attachments", []) for message in messages)

    if not chat_has_attachments:
        return [m if not isinstance(m, HumanMessage) else HumanMessage(content=m.content) for m in messages]

    supports_images = model.supports_images()

    """
    While dealing with models that don't support images,
    we transform the messages to include the image metadata in the prompt.
    Some models fail when the content is of the format { "type": "text", "text": "..." }
    Since we supply on image metadata to these models, we ensure that the content will be a string.
    Note - We might not need this if we decide to route requests with attachments always to ones with image support.
    """
    if not supports_images:
        transform_user_mesages: list[BaseMessage] = []
        for message in messages:
            attachments = message.additional_kwargs.get("attachments", [])
            if isinstance(message, HumanMessage):
                transform_user_mesages.append(
                    HumanMessage(content=get_images_polyfill_prompt(attachments, str(message.content)))
                )
            else:
                transform_user_mesages.append(message)
        return transform_user_mesages

    transformed_messages: list[BaseMessage] = []

    last_user_message_id_with_attachments = None

    """
    We need to find the last user message with attachments.
    For rest of the HumanMessages, we add the image metadata to the prompt.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            attachments = message.additional_kwargs.get("attachments", [])
            if len(attachments) > 0:
                last_user_message_id_with_attachments = message.additional_kwargs.get("message_id", "")
                break

    """
    When the code reaches here, we can assume that the content can be a
    dictionary of type { "type": "text", "text": "..." } or { "type": "image_url", "image_url": { "url": "..." } }
    """
    for message in messages:
        if not isinstance(message, HumanMessage):
            transformed_messages.append(message)
            continue

        attachments = message.additional_kwargs.get("attachments", [])
        message_id = message.additional_kwargs.get("message_id", "")

        if len(attachments) == 0:
            transformed_messages.append(message)
            continue

        content: list[str | dict[str, Any]] = []

        if last_user_message_id_with_attachments == message_id:
            attachment_tasks = []
            for attachment in attachments:
                attachment_tasks.append(generic_image_transformer(attachment))
            content.extend(r for r in await asyncio.gather(*attachment_tasks) if r is not Exception)
            content.append({"type": "text", "text": str(message.content)})
        else:
            content.append({"type": "text", "text": get_images_polyfill_prompt(attachments, str(message.content))})

        transformed_messages.append(HumanMessage(content=content))
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
