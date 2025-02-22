from collections.abc import AsyncIterator
from typing import Any

from google.cloud.aiplatform_v1beta1.types import GoogleSearchRetrieval
from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs.chat_generation import ChatGenerationChunk
from langchain_google_vertexai import ChatVertexAI


class GroundedVertexAI(ChatVertexAI):
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # Just some quirkiness from Google client.
        # For 1.5 it expects tool to be GoogleSearchRetrieval while 2.0 expects it to be GoogleSearch.
        # If we don't respect this, it throws a 400.
        if self.model_name.startswith("gemini-1.5"):
            kwargs["tools"] = [VertexTool(google_search_retrieval=GoogleSearchRetrieval())]
        else:
            kwargs["tools"] = [VertexTool(google_search=VertexTool.GoogleSearch())]

        async for chunk in super()._astream(messages=messages, stop=stop, run_manager=run_manager, **kwargs):
            if hasattr(chunk, "generation_info") and chunk.generation_info:
                if "grounding_metadata" in chunk.generation_info:
                    grounding_metadata = chunk.generation_info["grounding_metadata"]
                    if "grounding_chunks" in grounding_metadata:
                        citations = []
                        for grounding_chunk in grounding_metadata["grounding_chunks"]:
                            if "web" in grounding_chunk and "uri" in grounding_chunk["web"]:
                                citations.append(grounding_chunk["web"]["uri"])
                        if citations:
                            chunk.generation_info["citations"] = citations
            yield chunk
