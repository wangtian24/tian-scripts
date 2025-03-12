import asyncio
import base64
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from gcloud.aio.storage import Storage
from pillow_heif import register_heif_opener
from pydantic import BaseModel
from sqlmodel import select

from ypl.backend.attachments.image import downsize_image
from ypl.backend.attachments.upload import (
    create_attachment,
    get_bucket_name_and_root_folder,
    get_gcs_url,
    path_to_object,
    upload_original,
    upload_thumbnail,
)
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.utils.async_utils import create_background_task
from ypl.backend.utils.json import json_dumps
from ypl.db.attachments import Attachment

router = APIRouter()

register_heif_opener()


class AttachmentResponse(BaseModel):
    file_name: str
    attachment_id: str
    content_type: str


MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

SUPPORTED_MIME_TYPES_PATTERN = "image/.*|application/pdf"


@router.post("/file/upload", response_model=AttachmentResponse)
async def upload_file_route(file: UploadFile = File(...)) -> AttachmentResponse:  # noqa: B008
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

    attachment_gcs_bucket_path = settings.ATTACHMENT_BUCKET
    thumbnail_gcs_bucket_path = f"{attachment_gcs_bucket_path}/thumbnails"

    attachment_bucket, *attachment_path_parts = attachment_gcs_bucket_path.replace("gs://", "").split("/")
    thumbnail_bucket, *thumbnail_path_parts = thumbnail_gcs_bucket_path.replace("gs://", "").split("/")

    try:
        gcs_file_uuid = uuid4()
        gcs_thumbnail_uuid = uuid4()

        # Execute uploads in parallel
        gather_start = datetime.now()
        results = await asyncio.gather(
            create_background_task(
                upload_original(
                    gcs_file_uuid,
                    attachment_bucket,
                    attachment_path_parts,
                    file_content,
                    file.content_type,
                    file.filename,
                )
            ),
            create_background_task(
                upload_thumbnail(
                    gcs_thumbnail_uuid,
                    thumbnail_bucket,
                    thumbnail_path_parts,
                    file_content,
                    file.filename,
                )
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
            attachment = await create_attachment(
                attachment_bucket,
                attachment_path_parts,
                gcs_file_uuid,
                thumbnail_bucket,
                thumbnail_path_parts,
                gcs_thumbnail_uuid,
                metadata,
                file.content_type,
                file.filename,
            )
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


@dataclass
class DataUrlResponse:
    data_url: str


@router.get("/file/{attachment_id}/thumbnail", response_model=DataUrlResponse)
async def get_data_url_route(attachment_id: str) -> DataUrlResponse:
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


@dataclass
class SignedUrlResponse:
    signed_url: str
    file_id: UUID


@router.get("/attachment/signed-url")
async def get_signed_url_route() -> SignedUrlResponse:
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
async def create_attachment_route(request: CreateAttachmentRequest) -> Attachment:
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
