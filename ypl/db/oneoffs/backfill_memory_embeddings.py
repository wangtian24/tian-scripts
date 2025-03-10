import asyncio
import logging

import numpy as np
from more_itertools import chunked
from sqlalchemy import Connection, func, select
from sqlmodel import Session
from tqdm.asyncio import tqdm_asyncio as tqdm_asyncio

import ypl.db.all_models  # noqa
from ypl.backend.llm.embedding import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL, embed
from ypl.backend.utils.json import json_dumps
from ypl.db.embeddings import MemoryEmbedding
from ypl.db.memories import Memory


async def _embed(memories: list[Memory], sem: asyncio.Semaphore, session: Session) -> list[int]:
    async with sem:
        try:
            # Send all memories in a single batch.
            batch_embeddings = await embed(
                [m.memory_content for m in memories],  # type: ignore
                pad_to_length=DEFAULT_EMBEDDING_DIMENSION,
            )
            # Note that there may be multiple embeddings for a single memory, and since we sent multiple memories,
            # we now have a list of lists of embeddings and need a double loop.
            for memory_chunk_embeddings, memory in zip(batch_embeddings, memories, strict=True):
                for embedding in memory_chunk_embeddings:
                    me = MemoryEmbedding(
                        memory_id=memory.memory_id,
                        embedding=embedding,
                        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                    )
                    session.add(me)
            session.commit()
            return [len(embedding) for embedding in batch_embeddings]
        except Exception as e:
            log_dict = {
                "message": "Failed to embed memories",
                "memory_ids": [str(m.memory_id) for m in memories],
                "error": str(e),
            }
            logging.error(json_dumps(log_dict))
            return [0] * len(memories)


async def backfill_message_memory_embeddings(connection: Connection, max_memories: int | None = None) -> None:
    """Backfill memory embeddings for all memories."""
    num_parallel = 1
    memories_per_batch = 200
    with Session(connection) as session:
        base_query = (
            select(Memory)
            .outerjoin(
                MemoryEmbedding,
            )
            .where(
                MemoryEmbedding.memory_embedding_id.is_(None),  # type: ignore
                Memory.deleted_at.is_(None),  # type: ignore
                func.length(Memory.memory_content) > 0,
            )
            .order_by(Memory.created_at.desc())  # type: ignore
        ).distinct()

        if max_memories:
            base_query = base_query.limit(max_memories)

        # Get total count for progress bar
        count_stmt = select(func.count()).select_from(base_query.subquery())
        total_memories = session.exec(count_stmt).scalar_one()  # type: ignore
        logging.info(f"Total memories: {total_memories}")
        total_batches = total_memories // memories_per_batch
        logging.info(f"Total batches: {total_batches}")

        # Get the actual memories using the same base query
        memories = session.exec(base_query).scalars().all()  # type: ignore
        sem = asyncio.Semaphore(num_parallel)

        embedding_counts = await tqdm_asyncio.gather(
            *[_embed(memory_batch, sem, session) for memory_batch in chunked(memories, memories_per_batch)],
            total=total_batches,
        )
        # Flatten the list of counts, which is a list of lists (for each sublist, it lists the counts for each message).
        embedding_counts = [count for counts in embedding_counts for count in counts]
        counts, bins = np.histogram(embedding_counts, bins=range(max(embedding_counts) + 2))
        logging.info(f"Total embeddings: {sum(embedding_counts)}")
        logging.info("Distribution of embeddings per memory:")
        for i, count in enumerate(counts):
            if count > 0:
                logging.info(f"{bins[i]} embeddings: {count} messages")
