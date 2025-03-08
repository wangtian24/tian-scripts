import logging
import uuid
from collections.abc import Sequence

from langchain_core.messages import HumanMessage
from sqlmodel import select

from ypl.backend.db import get_async_session
from ypl.backend.llm.context import get_curated_chat_context
from ypl.backend.llm.judge import YuppMemoryExtractor
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, Turn
from ypl.db.memories import ChatMessageMemoryAssociation, Memory, MemorySource


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
            for mem_content in extracted_memories:
                memory_obj = Memory(
                    user_id=user_id,
                    memory_content=mem_content,
                    memory_source=memory_source,
                )
                session.add(memory_obj)
                await session.flush()
                # connect to the source message.
                session.add(ChatMessageMemoryAssociation(memory_id=memory_obj.memory_id, message_id=source_uuid))

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
