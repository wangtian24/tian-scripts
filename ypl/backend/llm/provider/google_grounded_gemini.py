from collections.abc import Iterator
from typing import Any

import google.generativeai.client as client
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.outputs.chat_generation import ChatGenerationChunk
from ypl.backend.llm.vendor_langchain_adapter import GeminiLangChainAdapter

"""
Create a Google-grounded Gemini model. See https://ai.google.dev/gemini-api/docs/grounding?lang=python
"""


class GoogleGroundedGemini(GeminiLangChainAdapter):
    @property
    def tools(self) -> list[str] | str | None:
        return "google_search_retrieval"

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.model._client is None:
            self.model._client = client.get_default_generative_client()

        request = self._prepare_request(messages, tools="google_search_retrieval")
        stream_resp = self.model._client.stream_generate_content(request=request)

        for chunk in stream_resp:
            if not chunk.candidates:
                continue

            content = chunk.candidates[0].content
            text = content.parts[0].text
            generation_info = dict(citations=None)

            if chunk.candidates[0].grounding_metadata and chunk.candidates[0].grounding_metadata.grounding_chunks:
                for gchunk in chunk.candidates[0].grounding_metadata.grounding_chunks:
                    if gchunk.web:
                        generation_info["citations"] = (generation_info["citations"] or []) + [gchunk.web.uri]  # type: ignore

            if chunk.candidates[0].finish_reason:
                generation_info["finish_reason"] = chunk.candidates[0].finish_reason.name

            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text), generation_info=generation_info)

            if run_manager:
                run_manager.on_llm_new_token(str(chunk.content), chunk=chunk)

            yield chunk
