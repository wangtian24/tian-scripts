import asyncio
import logging
import time
from contextvars import ContextVar
from typing import Any, Generic, Literal, TypeVar, cast

from async_lru import alru_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm.asyncio import tqdm_asyncio

from ypl.backend.prompts.quicktake import (
    SYSTEM_QUICKTAKE_PROMPT,
    USER_QUICKTAKE_PROMPT,
)
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import StopWatch
from ypl.utils import Delegator

logging.basicConfig(level=logging.WARNING)
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
OnErrorBehavior = Literal[
    "raise",  # Raise the exception to the caller.
    "use_error_value",  # Swallow the exception and return the defined error_value.
]

QT_CANT_ANSWER = "<CANT_ANSWER>"


class CantAnswerException(Exception):
    """Exception raised when the LLM explicitly returns "<CANT_ANSWER>"."""

    pass


class LLMLabeler(Generic[InputType, OutputType]):
    """
    Represents an LLM that takes in objects of type `InputType` and outputs a label of type `OutputType`.
    """

    cached = False

    def _maybe_truncate(self, input: str) -> str:
        if len(input) > 200:
            return input[:200] + "... (truncated)"
        return input

    def _get_log_info(self, input: InputType, start_time: float) -> dict[str, Any]:
        info = {
            "labeler": self.__class__.__name__,
            "elapsed_secs": time.time() - start_time,
        }
        if isinstance(input, str):
            info["input"] = self._maybe_truncate(input)
        return info

    def _log_error(self, input: InputType, prepared_input: dict[str, Any], e: Exception, start_time: float) -> None:
        info = self._get_log_info(input, start_time) | {
            "message": "Error labeling input",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        for k, v in prepared_input.items():
            info[f"prepared_input_{k}"] = self._maybe_truncate(v)

        logging.warning(json_dumps(info))

    def _log_success(self, input: InputType, output: BaseMessage, start_time: float) -> None:
        info = self._get_log_info(input, start_time) | {
            "message": "Labeled input",
            "output": str(output.content),
        }
        logging.info(json_dumps(info))

    def __init__(
        self, llm: BaseChatModel, timeout_secs: float = 5.0, on_error: OnErrorBehavior = "use_error_value"
    ) -> None:
        self.llm = self._prepare_llm(llm)
        self.asyncio_context: ContextVar = ContextVar("Coroutine local")
        self.timeout_secs = timeout_secs
        self.on_error = on_error

        if self.cached:
            self.alabel = alru_cache(maxsize=2048)(self.alabel)  # type: ignore[method-assign]

    @property
    def error_value(self) -> OutputType:
        """The value to return when an error occurs."""
        raise NotImplementedError

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        """Prepares the LLM for labeling. The default implementation is a no-op."""
        return llm

    def _prepare_input(self, input: InputType) -> dict[str, Any]:
        """Prepares the input to pass into the LLM's `.invoke` method."""
        raise NotImplementedError

    def _parse_output(self, output: BaseMessage) -> OutputType:
        """Parses the output of the LLM's `.invoke` method."""
        raise NotImplementedError

    async def _aparse_output(self, output: BaseMessage) -> OutputType:
        """Parses the output of the LLM's `.ainvoke` method. Defaults to calling `_parse_output`."""
        return self._parse_output(output)

    def _clean_output(self, output: BaseMessage) -> str:
        return str(output.content)

    def _validate_output(self, output: BaseMessage) -> None:
        """
        Checks if the output from LLM is a valid response. A subclass can override this method to
        throw an exception if the output indidates error or lack of answer. The default implementation is a no-op.
        Use case: quicktake labeler checks if the output is <CANT_ANSWER> and treats it as error.
        """
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
        retry=retry_if_exception_type((TimeoutError,)),
    )
    def label_full(self, input: InputType) -> tuple[OutputType, str]:
        """Labels the input."""
        try:
            start_time = time.time()
            prepared_input = self._prepare_input(input)
            output = self.llm.invoke(prepared_input)  # type: ignore
            self._validate_output(output)
            self._log_success(input, output, start_time)
            return self._parse_output(output), self._clean_output(output)
        except Exception as e:
            self._log_error(input, prepared_input, e, start_time)
            if self.on_error == "raise":
                raise e
            else:
                return self.error_value, ""

    def label(self, input: InputType) -> OutputType:
        return self.label_full(input)[0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
        retry=retry_if_exception_type((TimeoutError,)),
    )
    async def alabel_full(self, input: InputType) -> tuple[OutputType, str]:
        """Labels the input asynchronously."""
        try:
            start_time = time.time()
            async with asyncio.timeout(self.timeout_secs):
                prepared_input = self._prepare_input(input)
                output = await self.llm.ainvoke(prepared_input)  # type: ignore
                self._validate_output(output)
                self._log_success(input, output, start_time)
                return await self._aparse_output(output), self._clean_output(output)
        except Exception as e:
            self._log_error(input, prepared_input, e, start_time)
            if self.on_error == "raise":
                raise e
            else:
                return self.error_value, ""

    async def alabel(self, input: InputType) -> OutputType:
        with StopWatch(f"latency/{self.__class__.__name__}", auto_export=True):
            return (await self.alabel_full(input))[0]

    def batch_label(self, inputs: list[InputType]) -> list[OutputType | None]:
        """Labels a batch of inputs."""
        return [self.label(input) for input in inputs]

    def batch_label_full(self, inputs: list[InputType]) -> list[tuple[OutputType, str]]:
        """Labels a batch of inputs."""
        return [self.label_full(input) for input in inputs]

    async def abatch_label_full(
        self, inputs: list[InputType], num_parallel: int = 16
    ) -> list[tuple[OutputType, str] | None]:
        """Labels a batch of inputs asynchronously."""

        async def _do_label(input: InputType, sem: asyncio.Semaphore) -> tuple[OutputType, str] | None:
            async with sem:
                try:
                    return await self.alabel_full(input)
                except Exception as e:  # noqa catch-all errors for now
                    logging.exception(f"Error labeling input {input}: {e}")

                    if self.on_error == "raise":
                        raise e
                    else:
                        logging.warning(f"Error labeling input {input}: {e}")
                        return self.error_value, ""

        sem = asyncio.Semaphore(num_parallel)

        return await tqdm_asyncio.gather(*[_do_label(input, sem) for input in inputs])  # type: ignore

    async def abatch_label(self, inputs: list[InputType], num_parallel: int = 16) -> list[OutputType | None]:
        return [x[0] if x else None for x in await self.abatch_label_full(inputs, num_parallel)]


class MultiLLMLabeler(Delegator):
    """Applies multiple LLMLabelers to the same input and returns a dictionary of results."""

    def __init__(self, labelers: dict[str, LLMLabeler], *args: Any, **kwargs: Any) -> None:
        """
        Initializes the MultiLLMLabeler.

        Args:
            labelers: A dictionary mapping labeler names to LLMLabeler objects
            **kwargs: Additional keyword arguments to pass to the Delegator
        """
        super().__init__(delegates=labelers, **kwargs)


class QuickTakeRequest(BaseModel):
    chat_history: list[dict[str, Any]]
    prompt: str


class QuickTakeGenerator(LLMLabeler[HumanMessage, str]):
    """
    Writes a quicktake for a given conversation history and prompt.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        chat_history: list[BaseMessage],
        model_name: str,
        keep_role: str | None = "ai",
        user_quicktake_prompt: str = USER_QUICKTAKE_PROMPT,
        system_quicktake_prompt: str = SYSTEM_QUICKTAKE_PROMPT,
        **kwargs: Any,
    ) -> None:
        self.keep_role = keep_role
        self.chat_history = chat_history
        self.model_name = model_name
        self.user_quicktake_prompt = user_quicktake_prompt
        self.system_quicktake_prompt = system_quicktake_prompt
        super().__init__(llm, **kwargs)

    @property
    def error_value(self) -> str:
        return QT_CANT_ANSWER

    def _get_log_info(self, input: HumanMessage, start_time: float) -> dict[str, Any]:
        return {"model_name": self.model_name} | super()._get_log_info(input, start_time)

    def _get_prompt_template(self, message: HumanMessage) -> HumanMessage:
        if isinstance(message.content, str):
            return HumanMessage(content=self.user_quicktake_prompt)
        new_content: list[str | dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, str):  # We don't come across this case yet
                new_content.append(part)
                continue
            part = cast(dict[str, Any], part)
            if part["type"] != "text":
                new_content.append(part)
                continue
            new_content.append({"type": "text", "text": self.user_quicktake_prompt})
        return HumanMessage(content=new_content)

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        keep_roles = {"human", self.keep_role} if self.keep_role else {"human", "ai", "quicktake"}

        messages: list[BaseMessage] = [SystemMessage(content=self.system_quicktake_prompt)]

        for message in self.chat_history:
            if message.type not in keep_roles:
                continue
            if message.type == "quicktake":
                messages.append(AIMessage(content=message.content))
                continue
            messages.append(message)

        template = ChatPromptTemplate.from_messages([*messages, MessagesPlaceholder(variable_name="prompt")])

        return template | llm  # type: ignore

    def _prepare_input(self, prompt: HumanMessage) -> dict[str, Any]:
        return dict(prompt=[prompt])

    def _parse_output(self, output: BaseMessage) -> str:
        assert output.content is not None and isinstance(output.content, str)
        return str(output.content).strip()

    # Override
    def _validate_output(self, output: BaseMessage) -> None:
        if str(output.content).strip() == QT_CANT_ANSWER:
            raise CantAnswerException(f"{self.model_name} indicated it {QT_CANT_ANSWER}")
