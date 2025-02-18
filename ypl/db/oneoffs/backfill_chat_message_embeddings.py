import asyncio
import logging

import numpy as np
from sqlalchemy import Connection, func, select
from sqlmodel import Session, or_
from tqdm.asyncio import tqdm_asyncio as tqdm_asyncio

import ypl.db.all_models  # noqa
from ypl.backend.llm.embedding import DEFAULT_TOGETHER_MODEL, embed
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, CompletionStatus
from ypl.db.embeddings import ChatMessageEmbedding

EMBEDDING_MODEL = DEFAULT_TOGETHER_MODEL


async def _embed(message: ChatMessage, sem: asyncio.Semaphore, session: Session) -> int:
    if message.completion_status and message.completion_status.is_failure():
        return 0
    async with sem:
        try:
            embeddings = await embed(message.content, pad_to_length=1536)
            for embedding in embeddings:
                cme = ChatMessageEmbedding(
                    message_id=message.message_id,
                    embedding=embedding,
                    embedding_model_name=EMBEDDING_MODEL,
                )
                session.add(cme)
            session.commit()
            return len(embeddings)
        except Exception as e:
            log_dict = {
                "message": "Failed to embed chat message",
                "chat_message_id": str(message.message_id),
                "error": str(e),
            }
            logging.error(json_dumps(log_dict))
            return 0


async def backfill_chat_message_embeddings(connection: Connection, max_messages: int | None = None) -> None:
    """Backfill chat message embeddings for all chat messages."""
    num_parallel = 8
    with Session(connection) as session:
        base_query = (
            select(ChatMessage)
            .outerjoin(ChatMessageEmbedding)
            .where(
                ChatMessageEmbedding.embedding.is_(None),  # type: ignore
                ChatMessage.deleted_at.is_(None),  # type: ignore
                func.length(ChatMessage.content) > 0,
                or_(
                    ChatMessage.completion_status.is_(None),  # type: ignore
                    ChatMessage.completion_status.in_([CompletionStatus.SUCCESS, CompletionStatus.USER_ABORTED]),  # type: ignore
                ),
            )
            .order_by(ChatMessage.created_at.desc())  # type: ignore
        )

        if max_messages:
            base_query = base_query.limit(max_messages)

        # Get total count for progress bar
        count_stmt = select(func.count()).select_from(base_query.subquery())
        total_messages = session.exec(count_stmt).scalar_one()  # type: ignore
        logging.info(f"Total messages: {total_messages}")

        # Get the actual messages using the same base query
        messages = session.exec(base_query).scalars().all()  # type: ignore
        sem = asyncio.Semaphore(num_parallel)

        embedding_counts = await tqdm_asyncio.gather(
            *[_embed(message, sem, session) for message in messages],
            total=total_messages,
        )
        counts, bins = np.histogram(embedding_counts, bins=range(max(embedding_counts) + 2))
        logging.info(f"Total embeddings: {sum(embedding_counts)}")
        logging.info("Distribution of embeddings per message:")
        for i, count in enumerate(counts):
            if count > 0:
                logging.info(f"{bins[i]} embeddings: {count} messages")
