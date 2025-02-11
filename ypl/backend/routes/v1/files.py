import asyncio
import base64
import logging
import os
import re
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from gcloud.aio.storage import Storage
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import ExifTags, Image
from pillow_heif import register_heif_opener
from pydantic import BaseModel, SecretStr
from sqlmodel import select

from ypl.backend.db import get_async_session
from ypl.backend.prompts import IMAGE_DESCRIPTION_PROMPT
from ypl.backend.utils.json import json_dumps
from ypl.db.attachments import Attachment, TransientAttachment

client = ChatOpenAI(model="gpt-4o", api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")))

router = APIRouter()

register_heif_opener()


class AttachmentResponse(BaseModel):
    file_name: str
    attachment_id: str
    content_type: str


MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

SUPPORTED_MIME_TYPES_PATTERN = "image/.*|application/pdf"
EXIF_ORIENTATION_TAG = -1


def _get_exif_orientation_tag() -> int:
    global EXIF_ORIENTATION_TAG
    if EXIF_ORIENTATION_TAG == -1:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                EXIF_ORIENTATION_TAG = orientation
                break
    return EXIF_ORIENTATION_TAG


def maybe_rotate_image(image: Image.Image) -> Image.Image:
    tag = _get_exif_orientation_tag()
    try:
        exif = image._getexif()  # type: ignore
        if exif[tag] == 8:
            image = image.rotate(90, expand=True)
        elif exif[tag] == 3:
            image = image.rotate(180, expand=True)
        elif exif[tag] == 6:
            image = image.rotate(270, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass  # Image doesn't have EXIF data, or not rotated.
    return image


@router.post("/file/upload", response_model=AttachmentResponse)
async def upload_file(file: UploadFile = File(...)) -> AttachmentResponse:  # noqa: B008
    start_time = datetime.now()

    file_content = await file.read(MAX_FILE_SIZE_BYTES + 1)

    if len(file_content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds maximum limit")

    if not file.filename:
        log_dict = {"message": "Attachments: File name is required"}
        logging.warning(json_dumps(log_dict))
        raise HTTPException(status_code=400, detail="File name is required")
    if not file.content_type:
        log_dict = {"message": "Attachments: Content type is required", "file_name": file.filename}
        logging.warning(json_dumps(log_dict))
        raise HTTPException(status_code=400, detail="Content type is required")
    if not re.match(SUPPORTED_MIME_TYPES_PATTERN, file.content_type):
        log_dict = {
            "message": "Attachments: Unsupported file type",
            "file_name": file.filename,
            "content_type": file.content_type,
        }
        logging.warning(json_dumps(log_dict))
        raise HTTPException(status_code=400, detail="Unsupported file type")

    attachment_gcs_bucket_path = os.getenv("ATTACHMENT_BUCKET") or "gs://yupp-attachments/staging"
    thumbnail_gcs_bucket_path = f"{attachment_gcs_bucket_path}/thumbnails"

    attachment_bucket, *attachment_path_parts = attachment_gcs_bucket_path.replace("gs://", "").split("/")
    thumbnail_bucket, *thumbnail_path_parts = thumbnail_gcs_bucket_path.replace("gs://", "").split("/")

    try:

        async def upload_original(file_uuid: UUID) -> None:
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

        async def upload_thumbnail(thumbnail_uuid: UUID) -> tuple[int, int]:
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

        async def create_attachment(file_uuid: UUID, thumbnail_uuid: UUID, metadata: dict[str, Any]) -> Attachment:
            async with get_async_session() as session:
                thumbnail_url = (
                    f"gs://{thumbnail_bucket}/{'/'.join(thumbnail_path_parts)}/{thumbnail_uuid}"
                    if file.content_type and file.content_type.startswith("image/")
                    else ""
                )
                url = f"gs://{attachment_bucket}/{'/'.join(attachment_path_parts)}/{file_uuid}"
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

        gcs_file_uuid = uuid4()
        gcs_thumbnail_uuid = uuid4()

        # Execute uploads in parallel
        gather_start = datetime.now()
        results = await asyncio.gather(
            asyncio.create_task(upload_original(gcs_file_uuid)),
            asyncio.create_task(
                upload_thumbnail(gcs_thumbnail_uuid)
                if file.content_type and file.content_type.startswith("image/")
                else asyncio.sleep(0)
            ),
        )
        logging.info(
            json_dumps(
                {
                    "message": "Attachments: Parallel operations completed",
                    "start_time": gather_start,
                    "duration_ms": datetime.now() - gather_start,
                    "file_name": file.filename,
                }
            )
        )

        attachment = None
        result_itr = enumerate(results)

        for i, result in result_itr:
            if isinstance(result, Exception):
                if i == 1 and file.content_type and file.content_type.startswith("image/"):
                    log_dict = {
                        "message": f"Attachments: Error uploading thumbnail: {str(result)}",
                        "file_name": file.filename,
                        "content_type": file.content_type,
                    }
                    logging.warning(json_dumps(log_dict))
                    continue
                raise HTTPException(status_code=500, detail=str(result)) from result

        metadata = {"dimensions": results[1], "size": file.size}

        try:
            create_start = datetime.now()
            attachment = await create_attachment(gcs_file_uuid, gcs_thumbnail_uuid, metadata)
            logging.info(
                json_dumps(
                    {
                        "message": "Attachments: Attachment created in database",
                        "start_time": create_start,
                        "duration_ms": datetime.now() - create_start,
                        "file_name": file.filename,
                    }
                )
            )
        except Exception as e:
            log_dict = {"message": f"Attachments: Error creating attachment: {str(e)}"}
            logging.exception(json_dumps(log_dict))
            raise HTTPException(status_code=500, detail=str(e)) from e

        total_duration = datetime.now() - start_time
        logging.info(
            json_dumps(
                {
                    "message": "Attachments: File upload completed",
                    "start_time": start_time,
                    "total_duration_ms": total_duration,
                    "file_name": file.filename,
                    "attachment_id": str(attachment.attachment_id),
                }
            )
        )
        return AttachmentResponse(
            file_name=file.filename, attachment_id=str(attachment.attachment_id), content_type=file.content_type
        )

    except Exception as e:
        total_duration = datetime.now() - start_time
        logging.exception(
            json_dumps(
                {
                    "message": f"Attachments: Error uploading file: {str(e)}",
                    "total_duration_ms": total_duration,
                    "file_name": file.filename,
                }
            )
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


async def generate_image_description(file: TransientAttachment) -> dict[str, str]:
    start = datetime.now()
    image_bytes = file.file
    await asyncio.sleep(0)
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        messages: list[BaseMessage] = [
            SystemMessage(content=IMAGE_DESCRIPTION_PROMPT.format(file_name=file.filename)),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ]
            ),
        ]
        result = await client.ainvoke(messages)
        return {"description": str(result.content)}
    finally:
        logging.info(
            json_dumps(
                {
                    "message": "Attachments: Image description generated",
                    "duration_ms": datetime.now() - start,
                    "file_name": file.filename,
                }
            )
        )


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


@dataclass
class DataUrlResponse:
    data_url: str


@router.get("/file/{attachment_id}/thumbnail", response_model=DataUrlResponse)
async def get_data_url(attachment_id: str) -> DataUrlResponse:
    async with get_async_session() as session:
        result = await session.exec(select(Attachment).where(Attachment.attachment_id == attachment_id))
        attachment = result.one()

        if not attachment.content_type or not attachment.content_type.startswith("image/"):
            return DataUrlResponse(data_url="")

        thumbnail_url = attachment.thumbnail_url

        if not thumbnail_url:
            return DataUrlResponse(data_url="")

        if not thumbnail_url.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {thumbnail_url}")

        bucket_name, *path_parts = thumbnail_url.replace("gs://", "").split("/")
        if not bucket_name or not path_parts:
            raise ValueError(f"Invalid GCS path format: {thumbnail_url}")

        try:
            async with Storage() as async_client:
                image_bytes = await async_client.download(
                    bucket=bucket_name,
                    object_name=f"{'/'.join(path_parts)}",
                )
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                data_url = f"data:image/png;base64,{base64_image}"
                return DataUrlResponse(data_url=data_url)
        except Exception as e:
            logging.exception(f"Error downloading thumbnail: {str(e)}")
            return DataUrlResponse(data_url="")


THUMBNAILS_FOLDER = "thumbnails"
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


@dataclass
class SignedUrlResponse:
    signed_url: str
    file_id: UUID


@router.get("/attachment/signed-url")
async def get_signed_url() -> SignedUrlResponse:
    start = datetime.now()
    attachment_bucket, _ = get_bucket_name_and_root_folder()
    try:
        file_id = uuid4()
        blob_name = path_to_object(str(file_id))
        async with Storage() as async_client:
            bucket = async_client.get_bucket(attachment_bucket)
            blob = bucket.new_blob(blob_name=blob_name)
            signed_url = await blob.get_signed_url(http_method="PUT", expiration=120)  # Keep it valid only for 2 mins
            return SignedUrlResponse(signed_url=signed_url, file_id=file_id)
    except Exception as e:
        logging.exception(f"Error generating signed URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        logging.info(
            json_dumps(
                {
                    "message": "Attachments: Signed URL generated",
                    "duration_ms": datetime.now() - start,
                }
            )
        )


@dataclass
class CreateAttachmentRequest(BaseModel):
    file_name: str
    content_type: str
    file_id: UUID


@router.post("/attachment")
async def create_attachment(request: CreateAttachmentRequest) -> Attachment:
    blob = None
    if not request.content_type.startswith("image/"):
        raise ValueError(f"Unsupported content type {request.content_type}; must start with 'image/'")

    file_id = str(request.file_id)

    attachment_bucket, _ = get_bucket_name_and_root_folder()
    object_name = path_to_object(file_id)

    logging.info(f"Attachments: Creating attachment for file {file_id}")

    try:
        async with Storage() as async_client:
            blob = await async_client.download(bucket=attachment_bucket, object_name=object_name)
    except Exception as e:
        logging.exception(f"Attachments: Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not blob:
        raise HTTPException(status_code=500, detail="Failed to download file")

    if len(blob) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds maximum limit")

    thumbnail_start = datetime.now()
    thumbnail_path = path_to_object(file_id, THUMBNAILS_FOLDER)
    original_dimensions = None
    try:
        async with Storage() as async_client:
            thumbnail_bytes = await asyncio.to_thread(downsize_image, blob, (1024, 1024))
            logging.info(
                json_dumps(
                    {
                        "message": "Attachments: Thumbnail image resized",
                        "duration_ms": datetime.now() - thumbnail_start,
                        "file_name": request.file_name,
                        "file_id": file_id,
                    }
                )
            )
            await async_client.upload(
                bucket=attachment_bucket,
                object_name=thumbnail_path,
                file_data=thumbnail_bytes,
                content_type="image/png",
            )
            logging.info(
                json_dumps(
                    {
                        "message": "Attachments: Thumbnail upload completed",
                        "duration_ms": datetime.now() - thumbnail_start,
                        "file_name": request.file_name,
                    }
                )
            )
    except Exception as e:
        logging.exception(f"Attachments: Error uploading thumbnail: {str(e)}")
        raise e

    metadata = {"dimensions": original_dimensions, "size": len(blob)}

    thumbnail_url = get_gcs_url(file_id, THUMBNAILS_FOLDER)
    url = get_gcs_url(file_id)

    try:
        async with get_async_session() as session:
            attachment = Attachment(
                attachment_id=uuid4(),
                file_name=request.file_name,
                content_type=request.content_type,
                url=url,
                thumbnail_url=thumbnail_url,
                attachment_metadata=metadata,
            )
            session.add(attachment)
            await session.commit()
            await session.refresh(attachment)
            return attachment
    except Exception as e:
        logging.exception(f"Attachments: Error creating attachment: {str(e)}")
        raise e


def downsize_image(blob: bytes, size: tuple[int, int]) -> bytes:
    original_image = Image.open(BytesIO(blob))
    image = original_image.copy()
    image.thumbnail(size)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes.getvalue()
