import logging
import uuid
from collections.abc import Sequence

from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy import text
from sqlmodel import select

from ypl.backend.db import get_async_session
from ypl.backend.llm.context import get_curated_chat_context
from ypl.backend.llm.embedding import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL, embed
from ypl.backend.llm.judge import YuppMemoryExtractor
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, Turn
from ypl.db.embeddings import MemoryEmbedding
from ypl.db.memories import ChatMessageMemoryAssociation, Memory, MemorySource

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
