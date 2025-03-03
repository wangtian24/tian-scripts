import asyncio
import logging
import os
from collections.abc import Awaitable
from datetime import datetime
from io import BytesIO
from typing import Any
from uuid import UUID, uuid4

from fastapi import UploadFile
from gcloud.aio.storage import Storage
from PIL import Image
from sqlmodel import select
from ypl.backend.attachments.image import maybe_rotate_image
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.attachments import Attachment


async def upload_original(
    file_uuid: UUID,
    attachment_bucket: str,
    attachment_path_parts: list[str],
    file_content: bytes,
    file: UploadFile,
) -> None:
    start = datetime.now()
    try:
        async with Storage() as async_client:
            await async_client.upload(
                bucket=attachment_bucket,
                object_name=f"{'/'.join(attachment_path_parts)}/{file_uuid}",
                file_data=file_content,
                content_type=file.content_type,
            )
    finally:
        logging.info(
            json_dumps(
                {
                    "message": "Attachments: Original file upload completed",
                    "duration_ms": datetime.now() - start,
                    "file_name": file.filename,
                }
            )
        )


async def upload_thumbnail(
    thumbnail_uuid: UUID,
    thumbnail_bucket: str,
    thumbnail_path_parts: list[str],
    file_content: bytes,
    file: UploadFile,
) -> tuple[int, int]:
    start = datetime.now()
    try:
        async with Storage() as async_client:
            thumbnail_start = datetime.now()
            image = Image.open(BytesIO(file_content))
            image = maybe_rotate_image(image)  # type: ignore
            original_dimensions = image.size
            image.thumbnail((512, 512))
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes.seek(0)
            logging.info(
                json_dumps(
                    {
                        "message": "Attachments: Thumbnail image resized",
                        "duration_ms": datetime.now() - thumbnail_start,
                        "file_name": file.filename,
                    }
                )
            )
            await async_client.upload(
                bucket=thumbnail_bucket,
                object_name=f"{'/'.join(thumbnail_path_parts)}/{thumbnail_uuid}",
                file_data=image_bytes,
                content_type="image/png",
            )
            logging.info(
                json_dumps(
                    {
                        "message": "Attachments: Thumbnail upload completed",
                        "duration_ms": datetime.now() - start,
                        "file_name": file.filename,
                    }
                )
            )
            return original_dimensions
    except Exception as e:
        logging.exception(f"Error uploading thumbnail: {str(e)}")
        raise e


async def create_attachment(
    attachment_bucket: str,
    attachment_path_parts: list[str],
    attachment_uuid: UUID,
    thumbnail_bucket: str,
    thumbnail_path_parts: list[str],
    thumbnail_uuid: UUID,
    metadata: dict[str, Any],
    file: UploadFile,
) -> Attachment:
    async with get_async_session() as session:
        thumbnail_url = (
            f"gs://{thumbnail_bucket}/{'/'.join(thumbnail_path_parts)}/{thumbnail_uuid}"
            if file.content_type and file.content_type.startswith("image/")
            else ""
        )
        url = f"gs://{attachment_bucket}/{'/'.join(attachment_path_parts)}/{attachment_uuid}"
        attachment = Attachment(
            attachment_id=uuid4(),
            file_name=file.filename,
            content_type=file.content_type,
            url=url,
            thumbnail_url=thumbnail_url,
            attachment_metadata=metadata,
        )
        session.add(attachment)
        await session.commit()
        await session.refresh(attachment)
        return attachment


async def update_metadata(attachment: Attachment, image_description_task: Awaitable[Any]) -> None:
    await asyncio.sleep(0)
    start_time = datetime.now()
    try:
        # Generate image description
        image_description = await image_description_task

        # Update attachment with metadata
        async with get_async_session() as session:
            result = await session.exec(select(Attachment).where(Attachment.attachment_id == attachment.attachment_id))
            db_attachment = result.one()

            # Update metadata
            db_attachment.attachment_metadata = image_description
            await session.commit()

        logging.info(
            json_dumps(
                {
                    "message": "Attachments: Background metadata update completed",
                    "duration_ms": datetime.now() - start_time,
                    "file_name": attachment.file_name,
                    "attachment_id": str(attachment.attachment_id),
                }
            )
        )
    except Exception as e:
        logging.exception(
            json_dumps(
                {
                    "message": "Attachments: Error in background metadata update",
                    "error": str(e),
                    "duration_ms": datetime.now() - start_time,
                    "file_name": attachment.file_name,
                    "attachment_id": str(attachment.attachment_id),
                }
            )
        )


DEFAULT_ATTACHMENT_BUCKET = "gs://yupp-attachments/staging"


def get_bucket_name_and_root_folder() -> tuple[str, str]:
    # ATTACHMENT_BUCKET is of the format gs://bucket_name/path/to/root_folder
    # This function returns the bucket_name and the path_to_root_folder
    path_to_root_folder = os.getenv("ATTACHMENT_BUCKET") or DEFAULT_ATTACHMENT_BUCKET
    bucket_name, *path_parts = path_to_root_folder.replace("gs://", "").split("/")
    if not bucket_name or not path_parts:
        raise ValueError(f"Invalid GCS path format: {path_to_root_folder}")
    return bucket_name, os.path.join(*path_parts)


def path_to_object(object_id: str, folder_path: str = "") -> str:
    # object_id is the id of the object to be stored in GCS
    # folder_path is the path to the folder where the object is to be stored
    # This function returns the path to the object in GCS
    _, root_folder = get_bucket_name_and_root_folder()
    object_path = os.path.join(root_folder, folder_path, object_id)
    return object_path


def get_gcs_url(object_id: str, folder_path: str = "") -> str:
    # object_id is the id of the object to be stored in GCS
    # folder_path is the path to the folder where the object is to be stored
    # This function returns the GCS URL to the object
    attachment_bucket_gcs_path = os.getenv("ATTACHMENT_BUCKET") or DEFAULT_ATTACHMENT_BUCKET
    return os.path.join(attachment_bucket_gcs_path, folder_path, object_id)
