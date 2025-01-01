import asyncio
import logging
import os
from io import BytesIO
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from google.cloud import storage
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from ypl.backend.db import get_async_engine
from ypl.backend.utils.json import json_dumps
from ypl.db.attachments import Attachment

router = APIRouter()


@router.post("/file/upload", response_model=Attachment)
async def upload_file(file: UploadFile = File(...)) -> Attachment:  # noqa: B008
    attachment_gcs_bucket_path = os.getenv("ATTACHMENT_BUCKET") or "gs://yupp-attachments/staging"
    thumbnail_gcs_bucket_path = os.getenv("THUMBNAIL_BUCKET") or "gs://yupp-open/thumbnails/staging"

    attachment_bucket, *attachment_path_parts = attachment_gcs_bucket_path.replace("gs://", "").split("/")
    thumbnail_bucket, *thumbnail_path_parts = thumbnail_gcs_bucket_path.replace("gs://", "").split("/")

    if not file.filename:
        log_dict = {"message": "File name is required", "file_name": file.filename}
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=400, detail="File name is required")
    if not file.content_type:
        log_dict = {"message": "Content type is required", "file_name": file.filename}
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=400, detail="Content type is required")
    if not file.content_type.startswith("image/") and file.content_type != "application/pdf":
        log_dict = {"message": "Unsupported file type", "file_name": file.filename, "content_type": file.content_type}
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=400, detail="Unsupported file type")
    try:
        # Read file content once
        file_content = await file.read()
        file_io = BytesIO(file_content)

        async def upload_original(file_uuid: UUID) -> None:
            gcs_client = storage.Client()
            bucket = gcs_client.bucket(attachment_bucket)
            blob = bucket.blob(f"{'/'.join(attachment_path_parts)}/{file_uuid}")
            file_io.seek(0)
            blob.upload_from_file(file_io, content_type=file.content_type)

        async def upload_thumbnail(thumbnail_uuid: UUID) -> None:
            if file.content_type and file.content_type.startswith("image/"):
                gcs_client = storage.Client()
                bucket = gcs_client.bucket(thumbnail_bucket)
                thumbnail_blob = bucket.blob(f"{'/'.join(thumbnail_path_parts)}/{thumbnail_uuid}")
                image = Image.open(BytesIO(file_content))
                image.thumbnail((192, 192))
                image_bytes = BytesIO()
                image.save(image_bytes, format="PNG")
                image_bytes.seek(0)
                thumbnail_blob.upload_from_file(image_bytes, content_type="image/png")

        async def create_attachment(file_uuid: UUID, thumbnail_uuid: UUID) -> Attachment:
            async with AsyncSession(get_async_engine()) as session:
                thumbnail_url = (
                    f"https://storage.googleapis.com/{thumbnail_bucket}/{'/'.join(thumbnail_path_parts)}/{thumbnail_uuid}"
                    if file.content_type and file.content_type.startswith("image/")
                    else ""
                )
                attachment = Attachment(
                    attachment_id=uuid4(),
                    file_name=file.filename,
                    content_type=file.content_type,
                    url=f"gs://{attachment_bucket}/{'/'.join(attachment_path_parts)}/{file_uuid}",
                    thumbnail_url=thumbnail_url,
                )
                session.add(attachment)
                await session.commit()
                await session.refresh(attachment)
                return attachment

        gcs_file_uuid = uuid4()
        gcs_thumbnail_uuid = uuid4()

        # Execute uploads in parallel
        results = await asyncio.gather(
            upload_original(gcs_file_uuid),
            upload_thumbnail(gcs_thumbnail_uuid)
            if file.content_type and file.content_type.startswith("image/")
            else asyncio.sleep(0),
            create_attachment(gcs_file_uuid, gcs_thumbnail_uuid),
        )

        attachment = None
        result_itr = enumerate(results)
        for i, result in result_itr:
            if isinstance(result, Exception):
                if i == 1 and file.content_type and file.content_type.startswith("image/"):
                    log_dict = {
                        "message": f"Error uploading thumbnail: {str(result)}",
                        "file_name": file.filename,
                        "content_type": file.content_type,
                    }
                    logging.warning(json_dumps(log_dict))
                    continue
                raise HTTPException(status_code=500, detail=str(result)) from result
            elif isinstance(result, Attachment):
                attachment = result

        if attachment is None:
            raise HTTPException(status_code=500, detail="Failed to upload file")

        return attachment

    except Exception as e:
        log_dict = {"message": f"Error uploading file: {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
