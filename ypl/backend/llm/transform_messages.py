import asyncio
import base64
import logging
import os
from datetime import datetime
from typing import Any, Literal, TypedDict

import fitz
from gcloud.aio.storage import Storage
from langchain_core.messages import BaseMessage, HumanMessage

from ypl.backend.llm.db_helpers import PDF_ATTACHMENT_MIME_TYPE
from ypl.backend.llm.provider.provider_clients import get_model_provider_tuple
from ypl.backend.prompts import IMAGE_POLYFILL_PROMPT
from ypl.backend.utils.json import json_dumps
from ypl.db.attachments import Attachment
from ypl.db.language_models import Provider


class TransformOptions(TypedDict, total=False):
    use_signed_url: bool
    image_type: Literal["thumbnail", "original"]
    parse_pdf_locally: bool
    max_pdf_text: int | None


DEFAULT_OPTIONS: TransformOptions = {
    "use_signed_url": False,
    "image_type": "thumbnail",
    "parse_pdf_locally": False,
}


def get_bucket_and_object_name(url: str) -> tuple[str, str]:
    if not url or not url.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {url}")

    bucket_name, *path_parts = url.replace("gs://", "").split("/")
    if not bucket_name or not path_parts:
        raise ValueError(f"Invalid GCS path format: {url}")
    return bucket_name, os.path.join(*path_parts)


async def download_attachment(attachment: Attachment, transform_options: TransformOptions | None = None) -> bytes:
    start_time = datetime.now()
    gcs_path = get_download_url(attachment, transform_options)

    bucket_name, object_name = get_bucket_and_object_name(gcs_path)

    try:
        logging.info(f"Attachments: Downloading attachment {attachment.attachment_id} from {gcs_path}")
        async with Storage() as async_client:
            blob = await async_client.download(bucket=bucket_name, object_name=object_name)
            logging.info(
                f"Attachments: {attachment.attachment_id} - "
                f"GCS partial download took {datetime.now() - start_time} seconds"
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


PARSED_PDF_TEXT_TEMPLATE = """
The user has uploaded one or more PDF files, here is the extracted text from one of the files, enclosed between <PDF_START> and <PDF_END>. Please use the content below to help user answer their questions.
<PDF_START>
{text}
<PDF_END>
"""  # noqa: E501


async def generate_pdf_part_locally(attachment: Attachment, transform_options: TransformOptions) -> dict[str, Any]:
    file_bytes = await download_attachment(attachment, transform_options=None)
    max_text_len = transform_options.get("max_pdf_text", 32000) or 32000
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = []
    total_len = 0
    for page in doc:
        page_text = page.get_text()
        text.append(page_text)
        total_len += len(page_text)
        if total_len > max_text_len:
            break
    final_text = "".join(text)[:max_text_len]
    return {"type": "text", "text": PARSED_PDF_TEXT_TEMPLATE.format(text=final_text)}


async def generate_pdf_part(attachment: Attachment, provider: Provider) -> dict[str, Any]:
    file_bytes = await download_attachment(attachment, transform_options=None)
    if provider.name == "Anthropic":
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "data": base64.b64encode(file_bytes).decode("utf-8"),
                "media_type": attachment.content_type,
            },
        }
    elif provider.name == "Google":
        return {
            "type": "media",
            "mime_type": attachment.content_type,
            "data": base64.b64encode(file_bytes).decode("utf-8"),
        }
    raise ValueError(f"Unsupported provider: {provider.name}")


async def generate_image_part(attachment: Attachment, transform_options: TransformOptions) -> dict[str, Any]:
    if transform_options.get("use_signed_url", DEFAULT_OPTIONS["use_signed_url"]):
        url = await get_image_signed_url(attachment, transform_options)
    else:
        image_bytes = await download_attachment(attachment, transform_options)
        mime_type = (
            "image/png"
            if transform_options.get("image_type", DEFAULT_OPTIONS["image_type"]) == "thumbnail"
            else attachment.content_type
        )
        url = f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": url},
    }


async def generate_part(
    attachment: Attachment,
    provider: Provider,
    options: TransformOptions = DEFAULT_OPTIONS,
) -> dict[str, Any]:
    is_pdf = attachment.content_type.startswith(PDF_ATTACHMENT_MIME_TYPE)
    if is_pdf:
        if options.get("parse_pdf_locally", False):
            return await generate_pdf_part_locally(attachment, options)
        else:
            return await generate_pdf_part(attachment, provider)
    is_image = attachment.content_type.startswith("image/")
    if is_image:
        return await generate_image_part(attachment, options)
    raise ValueError(f"Unsupported attachment type: {attachment.content_type}")


async def transform_user_messages(
    messages: list[BaseMessage],
    model_name: str,
    options: TransformOptions = DEFAULT_OPTIONS,
) -> list[BaseMessage]:
    """
    Transform user messages by handling attachments and converting them to a format supported by the model.

    Args:
        messages: List of messages to transform
        model_name: Name of the model to transform messages for
        options: Options for transforming messages, including:
            - use_signed_url: Whether to use signed URLs for images
            - image_type: Type of image to use ('thumbnail' or original)

    Returns:
        List of transformed messages with attachments properly formatted for the model

    The function:
    1. Checks if the model supports images
    2. For messages without attachments, returns them unchanged
    3. For messages with attachments:
        - Generates image parts (either signed URLs or base64)
        - Combines image content with text content
        - Formats messages according to model requirements
    """
    model_provider = get_model_provider_tuple(model_name)
    if not model_provider:
        raise ValueError(f"No model-provider configuration found for: {model_name}")

    model = model_provider[0]

    all_attachments = [
        attachment for message in messages for attachment in message.additional_kwargs.get("attachments", [])
    ]
    filtered_attachments = []
    for attachment in all_attachments:
        if not model.supports_mime_type(attachment.content_type):
            log_dict = {
                "message": "Model does not support mime type. Skipping transformation for attachment.",
                "attachment_id": str(attachment.attachment_id),
                "mime_type": attachment.content_type,
                "model": model.name,
                "provider": model_provider[1].name,
            }
            logging.warning(json_dumps(log_dict))
            continue
        filtered_attachments.append(attachment)

    if not filtered_attachments:
        return [m if not isinstance(m, HumanMessage) else HumanMessage(content=m.content) for m in messages]

    transformed_messages: list[BaseMessage] = []

    attachment_id_to_content_dict: dict[str, dict[str, Any]] = {}
    attachment_tasks = [
        generate_part(attachment, provider=model_provider[1], options=options) for attachment in filtered_attachments
    ]
    results = await asyncio.gather(*attachment_tasks, return_exceptions=True)

    for attachment, result in zip(filtered_attachments, results, strict=True):
        if isinstance(result, BaseException):
            logging.warning(f"Attachments: skipping attachment: {attachment.attachment_id} - {str(result)}")
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


def get_download_url(attachment: Attachment, transform_options: TransformOptions | None = None) -> str:
    if transform_options is None:
        return attachment.url
    if transform_options.get("image_type", DEFAULT_OPTIONS["image_type"]) == "thumbnail":
        if not attachment.thumbnail_url:
            raise ValueError(f"Attachments: Thumbnail URL not found for attachment: {attachment.attachment_id}")
        return attachment.thumbnail_url
    return attachment.url


async def get_image_signed_url(attachment: Attachment, transform_options: TransformOptions) -> str:
    start_time = datetime.now()
    gcs_path = get_download_url(attachment, transform_options)
    try:
        if not gcs_path or not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        bucket_name, *path_parts = gcs_path.replace("gs://", "").split("/")
        if not bucket_name or not path_parts:
            raise ValueError(f"Invalid GCS path format: {gcs_path}")

        async with Storage() as async_client:
            bucket = async_client.get_bucket(bucket_name)
            blob = await bucket.get_blob(f"{'/'.join(path_parts)}")
            url = await blob.get_signed_url(expiration=120, http_method="GET")
            logging.info(
                f"Attachments: {attachment.attachment_id} - GCS signed url took {datetime.now() - start_time} seconds"
            )
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
        logging.warning(json_dumps(log_dict))
        raise RuntimeError(f"Failed to download file from GCS: {str(e)}") from e
