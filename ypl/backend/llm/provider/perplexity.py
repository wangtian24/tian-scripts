from collections.abc import Iterator
from typing import Any

from langchain_community.chat_models import ChatPerplexity
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.outputs.chat_generation import ChatGenerationChunk

"""
Overwrite ChatPerplexity to emit citations.
Support for citations is there inside _generate method but not in _stream method.
The _stream method emits citations only in the first chunk.
"""


class CustomChatPerplexity(ChatPerplexity):
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        default_chunk_class = AIMessageChunk
        emit_citations = True

        if stop:
            params["stop_sequences"] = stop
        stream_resp = self.client.chat.completions.create(model=params["model"], messages=message_dicts, stream=True)
        for chunk in stream_resp:
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            citations = chunk.get("citations", None)
            choice = chunk["choices"][0]
            chunk = self._convert_delta_to_message_chunk(choice["delta"], default_chunk_class)
            finish_reason = choice.get("finish_reason")
            generation_info = dict(citations=citations) if emit_citations and citations is not None else None
            if emit_citations:
                emit_citations = generation_info is None
            generation_info = dict(finish_reason=finish_reason) if finish_reason is not None else generation_info
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
