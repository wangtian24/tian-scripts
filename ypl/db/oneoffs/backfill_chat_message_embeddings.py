import asyncio
import logging

import numpy as np
from more_itertools import chunked
from sqlalchemy import Connection, func, select
from sqlmodel import Session, or_
from tqdm.asyncio import tqdm_asyncio as tqdm_asyncio

import ypl.db.all_models  # noqa
from ypl.backend.llm.embedding import DEFAULT_EMBEDDING_MODEL, embed
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, CompletionStatus
from ypl.db.embeddings import ChatMessageEmbedding

EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL


async def _embed(messages: list[ChatMessage], sem: asyncio.Semaphore, session: Session) -> list[int]:
    async with sem:
        try:
            embeddings = await embed([m.content for m in messages], pad_to_length=1536)
            for batched_embeddings, message in zip(embeddings, messages, strict=True):
                for embedding in batched_embeddings:
                    cme = ChatMessageEmbedding(
                        message_id=message.message_id,
                        embedding=embedding,
                        embedding_model_name=EMBEDDING_MODEL,
                    )
                    session.add(cme)
            session.commit()
            return [len(embedding) for embedding in embeddings]
        except Exception as e:
            log_dict = {
                "message": "Failed to embed chat messages",
                "chat_message_ids": [str(m.message_id) for m in messages],
                "error": str(e),
            }
            logging.error(json_dumps(log_dict))
            return [0] * len(messages)


async def backfill_chat_message_embeddings(connection: Connection, max_messages: int | None = None) -> None:
    """Backfill chat message embeddings for all chat messages."""
    num_parallel = 1
    messages_per_batch = 24
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
        total_batches = total_messages // messages_per_batch
        logging.info(f"Total batches: {total_batches}")

        # Get the actual messages using the same base query
        messages = session.exec(base_query).scalars().all()  # type: ignore
        sem = asyncio.Semaphore(num_parallel)

        embedding_counts = await tqdm_asyncio.gather(
            *[_embed(message_batch, sem, session) for message_batch in chunked(messages, messages_per_batch)],
            total=total_batches,
        )
        # Flatten the list of counts, which is a list of lists (for each sublist, it lists the counts for each message).
        embedding_counts = [count for counts in embedding_counts for count in counts]
        counts, bins = np.histogram(embedding_counts, bins=range(max(embedding_counts) + 2))
        logging.info(f"Total embeddings: {sum(embedding_counts)}")
        logging.info("Distribution of embeddings per message:")
        for i, count in enumerate(counts):
            if count > 0:
                logging.info(f"{bins[i]} embeddings: {count} messages")
