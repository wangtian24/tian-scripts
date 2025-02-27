import json
import logging
from uuid import UUID

from sqlalchemy import update
from sqlmodel import select

from ypl.backend.db import get_async_session
from ypl.db.attachments import Attachment


async def link_attachments(message_id: UUID, attachment_ids: list[UUID]) -> None:
    logging.info(f"Linking attachments {attachment_ids} to message {message_id}")
    try:
        async with get_async_session() as session:
            await session.exec(
                update(Attachment)
                .values(chat_message_id=message_id)
                .where(Attachment.attachment_id.in_(attachment_ids))  # type: ignore
            )
            await session.commit()

    except Exception as e:
        log_dict = {
            "message": "Failed to link attachments",
            "message_id": message_id,
            "attachment_ids": attachment_ids,
            "error": str(e),
        }
        logging.exception(json.dumps(log_dict))
        raise RuntimeError(f"Failed to link attachments: {str(e)}") from e


async def get_attachments(attachment_ids: list[UUID]) -> list[Attachment]:
    if not attachment_ids:
        return []
    try:
        async with get_async_session() as session:
            result = await session.exec(select(Attachment).where(Attachment.attachment_id.in_(attachment_ids)))  # type: ignore
            return list(result.all())
    except Exception as e:
        log_dict = {
            "message": "Failed to get attachments",
            "attachment_ids": attachment_ids,
            "error": str(e),
        }
        logging.exception(json.dumps(log_dict))
        raise RuntimeError(f"Failed to get attachments: {str(e)}") from e


def supports_image(mime_types: list[str]) -> bool:
    return any(mime_type.startswith("image/") for mime_type in mime_types)


def supports_pdf(mime_types: list[str]) -> bool:
    return any(mime_type.startswith("application/pdf") for mime_type in mime_types)
