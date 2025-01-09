import logging
from abc import abstractmethod
from functools import cached_property
from threading import Lock
from typing import Any

import google.generativeai as genai
import google.generativeai.client as client
import vertexai
from google.generativeai.types import content_types
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.outputs.chat_generation import ChatGeneration
from openai import AsyncOpenAI, OpenAI

from ypl.backend.llm.model_data_type import ModelInfo

GOOGLE_ROLE_MAP = dict(human="user", assistant="model", ai="model")
OPENAI_ROLE_MAP = dict(human="user", ai="assistant")


class VendorLangChainAdapter(BaseChatModel):
    model_info: ModelInfo
    model_config_: dict[str, Any]

    @abstractmethod
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError

    @property
    def _llm_type(self) -> str:
        return self.model_info.model


class GeminiLangChainAdapter(VendorLangChainAdapter):
    project_id: str
    region: str
    model: genai.GenerativeModel

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        GeminiLangChainAdapter is a wrapper around the Gemini LLM API. `model_config_` must contain `project_id` and
        `region` fields.
        """
        kwargs["project_id"] = kwargs["model_config_"].pop("project_id")
        kwargs["region"] = kwargs["model_config_"].pop("region")

        try:
            vertexai.init(project=kwargs["project_id"], location=kwargs["region"], api_key=kwargs["model_info"].api_key)
            genai.configure(api_key=kwargs["model_info"].api_key)
        except Exception as e:
            logging.error(f"Failed to initialize Vertex AI: {e}")
            raise ValueError("Failed to initialize Vertex AI") from e

        kwargs["model"] = genai.GenerativeModel(model_name=kwargs["model_info"].model)
        super().__init__(*args, **kwargs)

    @property
    def tools(self) -> list[str] | str | None:
        return self.model_config_.get("tools", None)  # type: ignore[no-any-return]

    @cached_property
    def sys_message_mutex(self) -> Lock:
        return Lock()

    def _prepare_request(self, messages: list[BaseMessage], **kwargs: Any) -> dict[str, Any]:
        system_message = next((message for message in messages if message.type == "system"), None)
        model_kwargs = {}

        try:
            self.sys_message_mutex.acquire()
            old_system_instruction = self.model._system_instruction

            if system_message:
                messages = [message for message in messages if message.type != "system"]
                self.model._system_instruction = content_types.to_content(system_message.content)
            else:
                self.sys_message_mutex.release()

            for key in ("safety_settings", "tools", "tool_config"):
                model_kwargs[key] = kwargs.get(key, getattr(self.model, f"_{key}", None))

            if self.tools:
                model_kwargs["tools"] = self.tools

            request = self.model._prepare_request(
                contents=[
                    genai.protos.Content(
                        role=GOOGLE_ROLE_MAP.get(message.type, message.type),
                        parts=[genai.protos.Part(text=str(message.content))],
                    )
                    for message in messages
                ],
                generation_config=self.model._generation_config,
                **model_kwargs,
            )
        finally:
            if self.sys_message_mutex.locked():
                self.model._system_instruction = old_system_instruction  # Reset the system instruction
                self.sys_message_mutex.release()

        return request  # type: ignore[no-any-return]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            if self.model._client is None:
                self.model._client = client.get_default_generative_client()

            gapic_response = self.model._client.generate_content(request=self._prepare_request(messages))

            try:
                return ChatResult(
                    generations=[
                        ChatGeneration(message=AIMessage(content=gapic_response.candidates[0].content.parts[0].text))
                    ]
                )
            except Exception as e:
                logging.exception(f"Error parsing Gemini response {gapic_response}: {e}")
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])
        except Exception as e:
            logging.exception(f"Error generating content: {e}")
            raise e


class OpenAILangChainAdapter(VendorLangChainAdapter):
    model: OpenAI
    async_model: AsyncOpenAI

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["model"] = OpenAI(api_key=kwargs["model_info"].api_key)
        kwargs["async_model"] = AsyncOpenAI(api_key=kwargs["model_info"].api_key)
        super().__init__(*args, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        for key, value in self.model_config_.items():
            kwargs.setdefault(key, value)

        response = self.model.chat.completions.create(
            messages=[dict(role=message.type, content=message.content) for message in messages],  # type: ignore[misc]
            model=self.model_info.model,
            **kwargs,
        )
        content = str(response.choices[0].message.content)

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        for key, value in self.model_config_.items():
            kwargs.setdefault(key, value)

        response = await self.async_model.chat.completions.create(
            messages=[
                dict(role=OPENAI_ROLE_MAP.get(message.type, message.type), content=message.content)  # type: ignore[misc]
                for message in messages
            ],
            model=self.model_info.model,
            **kwargs,
        )
        content = str(response.choices[0].message.content)

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
