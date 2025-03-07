import logging
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatGeneration, ChatResult
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages.base import BaseMessageChunk
from langchain_core.outputs.chat_generation import ChatGenerationChunk
from openai import AsyncOpenAI
from pydantic import PrivateAttr

IMAGE_GEN_TIMEOUT_SECS = 30.0
IMAGE_GEN_ERROR_MESSAGE = "\\<Image Generation Error\\>"


class ImageGenChatModel(BaseChatModel):
    """
    Base class for image generation chat models.

    The subclasses should implement async _agenerate_image()
    """

    _model_name: str = PrivateAttr()

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._model_name = model_name

    @property
    def _llm_type(self) -> str:
        return self._model_name

    @abstractmethod
    async def _agenerate_image(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates an image from the given prompt.

        kwargs are used by subclasses to pass more args to the underlying models.

        Returns the url of the generated image.
        In case of error, the implementation should raise an exception.
        """

    def _wrap_url(self, url: str | None) -> str:
        if not url:
            return IMAGE_GEN_ERROR_MESSAGE
        return f'\n\n<yapp class="image-gen">\n{{\n   "url": "{url}",\n   "caption": "Generated image"\n}}\n</yapp>'

    def _concat_messages(self, messages: list[BaseMessage]) -> str:
        return "\n\n".join([str(msg.content) for msg in messages])

    def _chat_generation_chunk(self, text: str) -> ChatGenerationChunk:
        return ChatGenerationChunk(message=BaseMessageChunk(type="text", content=text))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Only async version is supported")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        image_prompt = self._concat_messages(messages)
        url = await self._agenerate_image(image_prompt, **kwargs)
        if url and run_manager:
            await run_manager.on_llm_new_token(url)

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
        yield self._chat_generation_chunk("<think> Generating image...")

        try:
            # Use all the user prompts together as the image generation prompt.
            # This might not work well for some corner cases, need to improve if models support better context.
            image_prompt = self._concat_messages(messages)
            image_url = await self._agenerate_image(image_prompt, **kwargs)

            yield self._chat_generation_chunk("</think>")

            # Yield the final result
            if image_url and run_manager:
                await run_manager.on_llm_new_token(image_url)

            yield self._chat_generation_chunk(self._wrap_url(image_url))

        except Exception as e:
            logging.error(
                {
                    "message": f"Image generation error: {str(e)[:100]}...",
                    "model": self._model_name,
                    "error": str(e),
                }
            )
            raise e


class DallEChatModel(ImageGenChatModel):
    """
    The chat model adaptor for DALL-E models.

    ALLOWED_SIZES = ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
    """

    _async_client: AsyncOpenAI = PrivateAttr()

    def __init__(self, openai_api_key: str, **kwargs: Any):
        super().__init__(model_name="dall-e-3", **kwargs)
        self._async_client = AsyncOpenAI(api_key=openai_api_key, timeout=IMAGE_GEN_TIMEOUT_SECS)

    async def _agenerate_image(self, prompt: str, **kwargs: Any) -> str:
        # Call the DALL·E 3 API to generate an image.
        response = await self._async_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            # size="1792x1024", do mot select a size for now.
            quality="standard",
            n=1,
            response_format="url",
            **kwargs,
        )
        logging.info(
            {
                "message": "Dall-e-3 generated an image",
                "response": response.model_dump(),  # Includes revised prompt from the model.
            }
        )

        image_url = response.data[0].url
        if image_url:
            return image_url
        else:
            raise Exception("No image url returned from Dall-e-3")
