import logging
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import and_
from sqlmodel import Session, select, text

from ypl.backend.db import get_engine
from ypl.db.chats import ChatMessage, MessageType, Turn

# Constants for annotation keys
IS_REFUSAL_ANNOTATION_NAME = "is_refusal"
QUICK_RESPONSE_QUALITY_ANNOTATION_NAME = "quick_response_quality"


def check_missing_annotation_expr(annotations: dict, key: str) -> Any:
    """
    Check if an annotation is missing for a given key.

    Args:
        annotations: ChatMessage.annotations dict
        key: The annotation key to check

    Returns:
        SQLAlchemy expression for checking missing annotations
    """
    return (
        annotations.is_(None)  # type: ignore
        | (~annotations.has_key(key))  # type: ignore
        | (annotations[key].astext.is_(None))
    )


def get_turns_to_evaluate(
    session: Session,
    message_types: list[MessageType],
    annotation_name: str,
    max_num_turns: int,
    additional_conditions: list[Any] | None = None,
    additional_fields: list[Any] | None = None,
) -> Sequence[uuid.UUID] | Sequence[tuple[uuid.UUID, ...]]:
    """Get turn IDs that need evaluation based on missing annotations.

    Args:
        session: The database session
        message_types: List of message types to check for missing annotations
        annotation_name: The annotation key to check
        max_num_turns: Maximum number of turns to return
        additional_conditions: Additional conditions to add to the query
        additional_fields: Additional fields to return along with turn_id

    Returns:
        If no additional_fields, returns list of turn_ids.
        If additional_fields provided, returns list of tuples containing turn_id and additional fields.
    """
    fields = [Turn.turn_id]
    if additional_fields:
        fields.extend(additional_fields)

    query = (
        select(*fields)
        .join(ChatMessage)
        .where(
            ChatMessage.message_type.in_(message_types),  # type: ignore
            check_missing_annotation_expr(ChatMessage.annotations, annotation_name),
        )
        .order_by(Turn.created_at.desc())  # type: ignore
        .limit(max_num_turns)
    )

    if additional_conditions:
        query = query.where(and_(*additional_conditions))

    return session.exec(query).unique().all()


def update_message_annotations_in_chunks(update_values: list[dict[str, Any]], chunk_size: int = 100) -> None:
    """
    Update message annotations in chunks, keeping existing annotations.

    Args:
        update_values: List of dictionaries containing message_id, key, and value for annotation updates
        chunk_size: Size of chunks for batch processing
    """
    for i in range(0, len(update_values), chunk_size):
        start, end = i, min(i + chunk_size, len(update_values))
        chunk = update_values[start:end]
        with Session(get_engine()) as session:
            session.execute(
                text(
                    """
                    UPDATE chat_messages
                    SET annotations = COALESCE(annotations, '{}') || jsonb_build_object(:key, :value)
                    WHERE message_id = :message_id
                    """
                ),
                chunk,
            )
            session.commit()
            logging.info(f"Committed updates {start} to {end} out of {len(update_values)}")
    logging.info(f"Completed updating {len(update_values)} messages")
