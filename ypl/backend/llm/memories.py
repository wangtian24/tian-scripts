import logging
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from hdbscan import HDBSCAN
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy import text
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.db import get_async_session
from ypl.backend.llm.context import get_curated_chat_context
from ypl.backend.llm.embedding import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL, embed
from ypl.backend.llm.judge import MemoryCompactor, YuppMemoryExtractor
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, Turn
from ypl.db.embeddings import MemoryEmbedding
from ypl.db.memories import ChatMessageMemoryAssociation, Memory, MemorySource

# If a user has fewer than this number of memories, just let the compactor consolidate them all joinly.
# Otherwise, cluster them first, and apply the compactor to each cluster.
MIN_MEMORIES_TO_APPLY_CLUSTERING = 15

# The minimum number of memories to cluster together.
MIN_CLUSTER_SIZE = 2

# The maximum number of tokens to use for consolidating a single cluster of memories.
MAX_TOKENS = 1024

# Memories that are similar to existing memories at this threshold or higher are skipped.
MIN_MEMORY_SIMILARITY_TO_SKIP = 0.95


class MemorySimilarity(BaseModel):
    memory_id: uuid.UUID | None = None
    memory_content: str | None = None
    similarity: float | None = None


async def maybe_extract_memories(chat_id: uuid.UUID, turn_id: uuid.UUID, user_id: str) -> None:
    try:
        chat_context = await get_curated_chat_context(
            chat_id,
            use_all_models_in_chat_history=True,
            model="",
            current_turn_id=turn_id,
            include_current_turn=True,
            max_turns=1,
            max_message_length=4096,
            context_for_logging="maybe_extract_memories",
        )

        # Attempt to identify a source for the extracted memories.
        # The logic below uses the first human message it encounters.
        # It's correctness depends on max_turns=1 and not larger
        # than one, above.
        source_uuid = None
        source_msg = None
        # Zip the chat context to pair UUIDs with messages.
        for uuid, msg in zip(chat_context.uuids, chat_context.messages, strict=True):
            if isinstance(msg, HumanMessage):
                source_uuid = uuid
                source_msg = msg
                break
        if not source_msg:  # Do not proceed without a source.
            return

        labeler = YuppMemoryExtractor(await get_internal_provider_client("gemini-2.0-flash-001", max_tokens=512))
        extracted_memories = await labeler.alabel(chat_context.messages)
        logging.info(
            json_dumps(
                {
                    "message": "Extracted memories",
                    "turn_id": turn_id,
                    "extracted_memories": extracted_memories,
                    "user_id": user_id,
                }
            )
        )
        if not extracted_memories:
            return

        # ------------------------------------------------------------------
        # Populate memory objects and toss them into the database.
        # ------------------------------------------------------------------

        # Due to the prompt setup, which encourages the LLM to extract from
        # the user's turn in the conversation, we assume that the source is
        # USER_MESSAGE.
        memory_source = MemorySource.USER_MESSAGE

        async with get_async_session() as session:
            for memory_content in extracted_memories:
                embeddings = (await embed([memory_content], pad_to_length=DEFAULT_EMBEDDING_DIMENSION))[0]
                similar_memory = await get_most_similar_memory(user_id, embeddings)
                if similar_memory.similarity is not None and similar_memory.similarity > MIN_MEMORY_SIMILARITY_TO_SKIP:
                    logging.info(
                        json_dumps(
                            {
                                "message": "Skipping similar memory",
                                "memory_content": memory_content,
                                "similar_memory_content": similar_memory.memory_content,
                                "similarity": similar_memory.similarity,
                                "similar_memory_id": similar_memory.memory_id,
                            }
                        )
                    )
                    continue

                memory_obj = Memory(
                    user_id=user_id,
                    memory_content=memory_content,
                    memory_source=memory_source,
                )
                session.add(memory_obj)
                await session.flush()
                # connect to the source message.
                session.add(ChatMessageMemoryAssociation(memory_id=memory_obj.memory_id, message_id=source_uuid))
                await session.commit()
                logging.info(
                    json_dumps(
                        {
                            "message": "Memory stored",
                            "memory_content": memory_content,
                            "memory_id": memory_obj.memory_id,
                            "source_message_id": source_uuid,
                        }
                    )
                )

                for embedding in embeddings:
                    cme = MemoryEmbedding(
                        memory_id=memory_obj.memory_id,
                        embedding=embedding,
                        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                    )
                    session.add(cme)
                await session.commit()

    except Exception as e:
        logging.error(f"Error extracting memories: {e}")


async def get_memories(
    user_id: str | None = None,
    message_id: uuid.UUID | None = None,
    chat_id: uuid.UUID | None = None,
    limit: int = 10,
    offset: int = 0,
) -> tuple[Sequence[tuple[Memory, list[uuid.UUID]]], bool]:
    """Returns a list of memories with their associated message IDs, and an indicator of whether there are more rows."""
    async with get_async_session() as session:
        query = select(Memory).order_by(Memory.created_at.desc())  # type: ignore
        if user_id:
            query = query.where(Memory.user_id == user_id)
        if message_id:
            query = query.join(ChatMessageMemoryAssociation).where(
                ChatMessageMemoryAssociation.message_id == message_id
            )
        if chat_id:
            query = query.join(ChatMessageMemoryAssociation).join(ChatMessage).join(Turn).where(Turn.chat_id == chat_id)

        query = query.offset(offset).limit(limit + 1)
        results = await session.exec(query)
        memories = results.all()

        has_more_rows = len(memories) > limit
        if has_more_rows:
            memories = memories[:-1]

        # get the associated messages for the memories.
        memory_ids = [memory.memory_id for memory in memories]
        memory_to_messages: dict[uuid.UUID, list[uuid.UUID]] = {memory_id: [] for memory_id in memory_ids}
        if memory_ids:
            associations_query = select(
                ChatMessageMemoryAssociation.memory_id, ChatMessageMemoryAssociation.message_id
            ).where(ChatMessageMemoryAssociation.memory_id.in_(memory_ids))  # type: ignore
            associations_results = await session.exec(associations_query)
            for memory_id, message_id in associations_results:
                memory_to_messages[memory_id].append(message_id)

        return [(memory, memory_to_messages[memory.memory_id]) for memory in memories], has_more_rows


async def get_most_similar_memory(user_id: str, embeddings: list[list[float]]) -> MemorySimilarity:
    """Returns the most similar memory to one represented in the given embeddings list."""

    mem_sim = MemorySimilarity()
    if not embeddings:
        return mem_sim

    # Using just the first embedding in the list currently.
    embedding = embeddings[0]

    # (X <=> Y) is raw pgvector's cosine distance.
    # We normalize it to a similarity score between 0 and 1.
    async with get_async_session() as session:
        query = text(
            f"""
            SELECT
              memory_embeddings.memory_id,
              memories.memory_content,
              1 - (embedding <=> '{embedding}'::vector)/2 as similarity
            FROM memory_embeddings
            JOIN memories ON memory_embeddings.memory_id = memories.memory_id
            WHERE memories.user_id = '{user_id}'
            ORDER BY similarity DESC
            LIMIT 1
            """
        )

        result = await session.execute(query)
        similar_memory = result.one_or_none()

        if similar_memory:
            mem_sim.memory_id = similar_memory[0]
            mem_sim.memory_content = similar_memory[1]
            mem_sim.similarity = similar_memory[2]

        return mem_sim


@dataclass  # pydantic doesn't work well with numpy arrays
class MemoryWithEmbeddingAndMessageId:
    memory_id: uuid.UUID
    memory_content: str
    embedding: np.ndarray
    message_id: uuid.UUID


def cluster_memories(
    memories: list[MemoryWithEmbeddingAndMessageId],
) -> list[list[MemoryWithEmbeddingAndMessageId]]:
    """Cluster memories based on embedding similarity; each cluster is a list of related memories."""
    if len(memories) < MIN_MEMORIES_TO_APPLY_CLUSTERING:
        # Just put everything in one cluster.
        return [memories]

    embeddings = np.vstack([memory.embedding for memory in memories])
    clusterer = HDBSCAN(metric="l2", min_cluster_size=MIN_CLUSTER_SIZE)
    cluster_labels = clusterer.fit_predict(embeddings)

    clusters = defaultdict(list)
    last_label = len(cluster_labels)
    for memory, label in zip(memories, cluster_labels, strict=True):
        if label == -1:
            # This means the memory did not cluster with any other memories, assign a new cluster.
            clusters[last_label] = [memory]
            last_label += 1
        else:
            clusters[label].append(memory)

    return list(clusters.values())


async def get_memories_with_embeddings(session: AsyncSession, user_id: str) -> list[MemoryWithEmbeddingAndMessageId]:
    query = (
        select(
            Memory.memory_id,
            Memory.memory_content,
            MemoryEmbedding.embedding,
            ChatMessageMemoryAssociation.message_id,
        )
        .join(ChatMessageMemoryAssociation)
        .join(MemoryEmbedding)
        .where(Memory.user_id == user_id, Memory.deleted_at.is_(None))  # type: ignore
    )
    results = await session.exec(query)
    return [
        MemoryWithEmbeddingAndMessageId(memory_id=row[0], memory_content=row[1], embedding=row[2], message_id=row[3])  # type: ignore
        for row in results.all()
    ]


async def consolidate_memories(user_id: str) -> None:
    """Consolidate the memories for a user by merging similar one."""
    try:
        async with get_async_session() as session:
            memories = await get_memories_with_embeddings(session, user_id)
            compactor = MemoryCompactor(
                llm=await get_internal_provider_client("gemini-2.0-flash-001", max_tokens=MAX_TOKENS),
                timeout_secs=15,
            )

            # The new rows that should be added to the memories table.
            new_memory_objs = []
            # The new rows that should be added to the chat_message_memory_associations table.
            new_memory_associations = []
            # The new rows that should be added to the memory_embeddings table.
            new_memory_embeddings_objs = []

            consolidated_memories = []
            memory_ids_to_delete = []
            clusters = cluster_memories(memories)
            # TODO(gilad): this information may be too detailed; after debug period, ok to remove.
            log_dict = {
                "message": "Consolidating memories",
                "user_id": user_id,
                "num_memories_to_consolidate": len(memories),
                "num_consolidated_memories": 0,
                "memory_clusters": [],
            }
            # TODO(gilad): consider processing in parallel; need to make sure it doesn't trigger rate limits.
            for cluster in clusters:
                memories_to_consolidate = [memory.memory_content for memory in cluster]
                message_ids = [memory.message_id for memory in cluster]
                # Used to track information we want to log.
                cluster_info = {"memories": memories_to_consolidate, "message_ids": message_ids}
                if len(cluster) == 1:
                    # Nothing to consolidate; keep the single memory.
                    cluster_info["consolidated"] = False
                    log_dict["memory_clusters"].append(cluster_info)  # type: ignore
                    continue

                # From here on, we have a cluster of more than one memory to consolidate.
                cluster_info["consolidated"] = True
                # Track the memories to delete later.
                memory_ids_to_delete.extend([memory.memory_id for memory in cluster])

                consolidated_memories = await compactor.alabel(memories_to_consolidate)
                cluster_info["consolidated_memories"] = consolidated_memories
                log_dict["memory_clusters"].append(cluster_info)  # type: ignore
                for consolidated_memory in consolidated_memories:
                    # Create the memory, its associations to messages, and embeddings.
                    new_memory_id = uuid.uuid4()
                    new_memory_objs.append(
                        Memory(
                            memory_id=new_memory_id,
                            user_id=user_id,
                            memory_content=consolidated_memory,
                            memory_source=MemorySource.CONSOLIDATED_MEMORY,
                        )
                    )
                    new_memory_associations.extend(
                        [ChatMessageMemoryAssociation(memory_id=new_memory_id, message_id=m_id) for m_id in message_ids]
                    )
                    new_embeddings = (await embed([consolidated_memory], pad_to_length=DEFAULT_EMBEDDING_DIMENSION))[0]
                    for embedding in new_embeddings:
                        new_memory_embeddings_objs.append(
                            MemoryEmbedding(
                                memory_id=new_memory_id,
                                embedding=embedding,
                                embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                            )
                        )

            log_dict["num_consolidated_memories"] = len(new_memory_objs)

            # Clean up the memories that were consolidated, and their associations and embeddings.
            delete_associations_query = delete(ChatMessageMemoryAssociation).where(
                ChatMessageMemoryAssociation.memory_id.in_(memory_ids_to_delete),  # type: ignore
            )
            await session.exec(delete_associations_query)  # type: ignore
            await session.flush()

            delete_embeddings_query = delete(MemoryEmbedding).where(MemoryEmbedding.memory_id.in_(memory_ids_to_delete))  # type: ignore
            await session.exec(delete_embeddings_query)  # type: ignore
            await session.flush()

            delete_memories_query = delete(Memory).where(Memory.memory_id.in_(memory_ids_to_delete))  # type: ignore
            await session.exec(delete_memories_query)  # type: ignore
            await session.flush()

            # Finally, add the new memories, associations, and embeddings.
            session.add_all(new_memory_objs)
            await session.flush()  # need to be added first for the next two adds to work.

            session.add_all(new_memory_associations)
            session.add_all(new_memory_embeddings_objs)
            await session.commit()

            logging.info(json_dumps(log_dict))

    except Exception as e:
        logging.error(f"Error consolidating memories: {e}")
