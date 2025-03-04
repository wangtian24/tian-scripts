import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatGeneration, ChatResult
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages.base import BaseMessageChunk
from langchain_core.outputs.chat_generation import ChatGenerationChunk
from openai import OpenAI
from pydantic import PrivateAttr

IMAGE_GEN_TIMEOUT_SECS = 30.0
IMAGE_GEN_ERROR_MESSAGE = "\\<Image Generation Error\\>"


class DallEChatModel(BaseChatModel):
    """
    The chat model adaptor for DALL-E models.

    ALLOWED_SIZES = ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
    """

    _client: OpenAI = PrivateAttr()

    def __init__(self, openai_api_key: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = OpenAI(api_key=openai_api_key, timeout=IMAGE_GEN_TIMEOUT_SECS)

    @property
    def _llm_type(self) -> str:
        return "dalle"

    def _generate_image(self, messages: list[BaseMessage], **kwargs: Any) -> str | None:
        # For simplicity, we take the content of the first message as our prompt.
        combined_msg = "\n\n".join([str(msg.content) for msg in messages])
        try:
            # Call the DALL·E 3 API to generate an image.
            response = self._client.images.generate(
                model="dall-e-3",
                prompt=combined_msg,
                size="1792x1024",  # TODO: this can adapt to user's prompt instructions
                quality="standard",
                n=1,
                response_format="url",
                **kwargs,
            )
            # Get the resulting image URL (or you could modify this to return base64 data)
            return response.data[0].url
        except Exception as e:
            logging.error(json.dumps({"message": f"Image generation error: {str(e)[:100]}...", "details": str(e)}))
            return None

    def _wrap_url(self, url: str | None) -> str:
        if not url:
            return IMAGE_GEN_ERROR_MESSAGE
        return f'\n\n<yapp class="image">\n{{\n   "url": "{url}",\n   "caption": "Generated image"\n}}\n</yapp>'

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        url = self._generate_image(messages, **kwargs)
        if url and run_manager:
            run_manager.on_llm_new_token(url)

        content = self._wrap_url(url)
        return ChatResult(
            generations=[ChatGeneration(message=BaseMessage(type="text", content=content, role="assistant"))]
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # First yield a thinking message
        think_start = BaseMessageChunk(type="text", content="<think>Generating your image...")
        yield ChatGenerationChunk(message=think_start)

        # Then generate the actual image using the same logic as _generate
        image_url = await asyncio.to_thread(self._generate_image, messages, **kwargs)

        # Close the thinking message after the picture is generated, so we keep the thinking section open on UI.
        think_end = BaseMessageChunk(type="text", content="</think>")
        yield ChatGenerationChunk(message=think_end)

        # Yield the final result
        if image_url and run_manager:
            await run_manager.on_llm_new_token(image_url)
        content = self._wrap_url(image_url)
        yield ChatGenerationChunk(message=BaseMessageChunk(type="text", content=content))
