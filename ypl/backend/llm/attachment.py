import json
import logging
from uuid import UUID

from sqlalchemy import update

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
