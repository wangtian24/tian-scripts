import logging
import uuid

from langchain_core.messages import HumanMessage

from ypl.backend.db import get_async_session
from ypl.backend.llm.chat import get_curated_chat_context
from ypl.backend.llm.judge import YuppMemoryExtractor
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.utils.json import json_dumps
from ypl.db.memories import Memory, MemorySource


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
                    source_message_id=source_uuid,
                    # Eventually, we will store the embedding.
                    # content_pgvector=some_embedding_vector,
                    # tags=['extracted'],
                    # agent_language_model_id=some_llm_id,  # Source is human for now.
                )
                session.add(memory_obj)

            await session.commit()

    except Exception as e:
        logging.error(f"Error extracting memories: {e}")
