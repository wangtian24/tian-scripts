import logging
import os
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import fal_client
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


class FalAIImageGenModel(ImageGenChatModel):
    """
    Base class for Fal AI Image Generation models.

    Guide for adding a new model to the language model table:
      - Use fal-ai model name for `name` field (e.g. "fal-ai/flux/schnell")
      - Set `internal_name` to shorter name used in logging etc. (e.g. "flux-schnell")
      - Set `label` to the model name, close to how it is known (e.g. "FLUX.1 [schnell]")
      - Set `external_model_url` to the model page on fal.ai (e.g. https://fal.ai/models/fal-ai/flux/schnell)
      - Set `parameters` to pass additional parameters. These are passed to 'arguments' option in `subscribe` method.
         - To help the readers, include "comment" to explain the parameters.
      - See example insert statements below for language_models table.
      - After this, add a new row for the model in `routing_rules` table. See example insert statements below.

    Sample insert statement into language_models table:
    -----------------------
        INSERT INTO language_models (
            language_model_id,
            name,
            internal_name,
            label,
            license,
            created_at,
            status,
            creator_user_id,
            provider_id,
            external_model_info_url,
            parameters,
            is_image_generation
        )
        VALUES (
            gen_random_uuid(),
            'fal-ai/flux-pro/new',
            'flux-pro',
            'FLUX.1 [pro]',
            'unknown',
            now(),
            'ACTIVE',
            'c7144895-6e3b-4d71-b1ad-24c83d50d73a',
            (SELECT provider_id from providers WHERE name = 'FalAI'),
            'https://fal.ai/models/fal-ai/flux-pro/new',
            $${
            "kwargs": {
                "num_inference_steps": 40
            },
            "comment": "num_inference_steps is set to 40 here. The default is 28 and max is 50."
            }$$,
            true
        );
    -----------------------

    Sample insert statement into routing_rules table:
    -----------------------
        INSERT INTO routing_rules (
            created_at,
            routing_rule_id,
            is_active,
            z_index,
            source_category,
            destination,
            target,
            probability
        )
        VALUES(
            now(),
            gen_random_uuid(),
            true,
            2000000,
            'Image Generation',
            'FalAI/flux-schnell',
            'ACCEPT',
            1.0
        );
    -----------------------
    """

    _async_client: fal_client.AsyncClient = PrivateAttr()
    _fal_model_name: str = PrivateAttr()
    _extra_options: dict[str, Any] = PrivateAttr()

    def __init__(self, model_name: str, fal_model_name: str, api_key: str = "", **kwargs: Any):
        """
        Args:
            model_name: This is the Yupp model to use in logging etc. E.g. "flux-pro"
            fal_model_name: The full name of Fal AI model. E.g. "fal-ai/flux-pro/v1.1-ultra"
            api_key: The API key to use.
            kwargs: Additional arguments are merged into the `arguments` option for the `subscribe` method.
        """
        super().__init__(model_name=fal_model_name, **kwargs)
        key = api_key or os.environ["FAL_AI_API_KEY"]
        self._fal_model_name = fal_model_name
        self._async_client = fal_client.AsyncClient(key=key)
        self._extra_options = kwargs

    async def _agenerate_image(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate an image from the given prompt.
        """

        # More documentation: https://docs.fal.ai/clients/python
        arguments = (
            {
                "prompt": prompt,
                "num_images": 1,
                "enable_safety_checker": True,  # Same as default. Setting it just for clarity.
            }
            | self._extra_options  # Options from the language model table for this model.
            | kwargs  # callers can add or override options
        )

        response: dict[str, Any] = await self._async_client.subscribe(
            application=self._fal_model_name,
            arguments=arguments,
            on_queue_update=None,  # FYI. We can log progress updates if we want, but not all images got them.
        )

        logging.info(
            {
                "message": f"{self._model_name} generated an image",
                "response": response,  # Includes revised prompt from the model.
                "fal_model_name": self._fal_model_name,
            }
        )
        if not response.get("images"):
            raise Exception(f"No image url returned from {self._model_name} (Fal model {self._fal_model_name})")

        return str(response["images"][0]["url"])
