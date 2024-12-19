import logging
from abc import abstractmethod
from typing import Any

import vertexai
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.outputs.chat_generation import ChatGeneration
from openai import AsyncOpenAI, OpenAI
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel

from ypl.backend.llm.chat import ModelInfo

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
    model: GenerativeModel

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        GeminiLangChainAdapter is a wrapper around the Gemini LLM API. `model_config_` must contain `project_id` and
        `region` fields.
        """
        kwargs["project_id"] = kwargs["model_config_"].pop("project_id")
        kwargs["region"] = kwargs["model_config_"].pop("region")

        try:
            vertexai.init(project=kwargs["project_id"], location=kwargs["region"], api_key=kwargs["model_info"].api_key)
        except Exception as e:
            logging.error(f"Failed to initialize Vertex AI: {e}")
            raise ValueError("Failed to initialize Vertex AI") from e

        kwargs["model"] = GenerativeModel(model_name=kwargs["model_info"].model)
        super().__init__(*args, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        goog_messages = [
            dict(content=str(message.content), role=GOOGLE_ROLE_MAP.get(message.type, message.type))
            for message in messages[:-1]
        ]

        chat = self.model.start_chat(history=goog_messages)
        response = chat.send_message(
            messages[-1].content,
            generation_config=GenerationConfig(
                **self.model_config_,
            ),
        )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response.text))])


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
