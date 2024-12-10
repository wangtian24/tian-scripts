import asyncio
import logging
import random
from contextvars import ContextVar
from typing import Any, Generic, Literal, TypeVar

from datasets import Dataset, load_dataset
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm.asyncio import tqdm_asyncio

from ypl.backend.llm.embedding import cached_get_database
from ypl.backend.prompts import (
    QUICKTAKE_SUMMARIZING_PROMPT_TEMPLATE_1,
    QUICKTAKE_SUMMARIZING_PROMPT_TEMPLATE_2,
    WILDCHAT_REALISM_PROMPT_TEMPLATE,
)

logging.basicConfig(level=logging.WARNING)
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
OnErrorBehavior = Literal[
    "raise",  # Raise the exception to the caller.
    "use_error_value",  # Swallow the exception and return the defined error_value.
]


class StopMultistepProcessing(Exception):
    """
    Exception raised to stop the multistep processing of an input.
    """

    pass


class LLMLabeler(Generic[InputType, OutputType]):
    """
    Represents an LLM that takes in objects of type `InputType` and outputs a label of type `OutputType`.
    """

    def __init__(
        self, llm: BaseChatModel, timeout_secs: float = 5.0, on_error: OnErrorBehavior = "use_error_value"
    ) -> None:
        self.llm = self._prepare_llm(llm)
        self.asyncio_context: ContextVar = ContextVar("Coroutine local")
        self.timeout_secs = timeout_secs
        self.on_error = on_error

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
        retry=retry_if_exception_type((TimeoutError,)),
    )
    def label(self, input: InputType) -> OutputType:
        """Labels the input."""
        try:
            prepared_input = self._prepare_input(input)
            output = self.llm.invoke(prepared_input)  # type: ignore

            return self._parse_output(output)
        except Exception as e:
            if self.on_error == "raise":
                raise e
            else:
                logging.warning(f"Error labeling input {input}: {e}")
                return self.error_value

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
        retry=retry_if_exception_type((TimeoutError,)),
    )
    async def alabel(self, input: InputType) -> OutputType:
        """Labels the input asynchronously."""
        try:
            async with asyncio.timeout(self.timeout_secs):
                prepared_input = self._prepare_input(input)
                output = await self.llm.ainvoke(prepared_input)  # type: ignore

                return await self._aparse_output(output)
        except Exception as e:
            if self.on_error == "raise":
                raise e
            else:
                logging.warning(f"Error labeling input {input}: {e}")
                return self.error_value

    def batch_label(self, inputs: list[InputType]) -> list[OutputType | None]:
        """Labels a batch of inputs."""
        return [self.label(input) for input in inputs]

    async def abatch_label(self, inputs: list[InputType], num_parallel: int = 16) -> list[OutputType | None]:
        """Labels a batch of inputs asynchronously."""

        async def _do_label(input: InputType, sem: asyncio.Semaphore) -> OutputType | None:
            async with sem:
                try:
                    return await self.alabel(input)
                except Exception as e:  # noqa catch-all errors for now
                    logging.exception(f"Error labeling input {input}: {e}")

                    if self.on_error == "raise":
                        raise e
                    else:
                        logging.warning(f"Error labeling input {input}: {e}")
                        return self.error_value

        sem = asyncio.Semaphore(num_parallel)

        return await tqdm_asyncio.gather(*[_do_label(input, sem) for input in inputs])  # type: ignore


class MultistepLLMLabeler(Generic[InputType, OutputType]):
    """
    Represents an LLM that takes in objects of type `InputType` and outputs a label of type `OutputType`, accomplishing
    this task using one or more sequential calls to the LLM. Subclasses should additionally implement the
    `_prepare_intermediate` method, which performs a single step of index `i` of the labeling process.

    Intermediate states must be a `dict[str, Any]` object, which will be passed to the next step.
    """

    def __init__(
        self, llm: BaseChatModel, timeout_secs: float = 15.0, on_error: OnErrorBehavior = "use_error_value"
    ) -> None:
        self.llms = [self._prepare_llm(step_no, llm) for step_no in range(1, self.num_steps + 1)]
        self.asyncio_context: ContextVar = ContextVar("Coroutine local")
        self.timeout_secs = timeout_secs
        self.on_error = on_error

    @property
    def num_steps(self) -> int:
        """The number of steps in the labeling process."""
        raise NotImplementedError

    @property
    def error_value(self) -> OutputType:
        """The value to return when an error occurs."""
        raise NotImplementedError

    def _prepare_llm(self, step_no: int, llm: BaseChatModel) -> BaseChatModel:
        """
        Prepares the LLM for labeling. The default implementation is a no-op. `step_no` is the one-based step number.
        """
        return llm

    def _prepare_initial_input(self, input: InputType) -> dict[str, Any]:
        """Prepares the input to pass into the LLM's `.invoke` method."""
        raise NotImplementedError

    def _prepare_intermediate(self, step_no: int, base_message: BaseMessage, state: dict[str, Any]) -> dict[str, Any]:
        """Extracts the next state from the LLM's output."""
        raise NotImplementedError

    def _parse_final_output(self, state: dict[str, Any]) -> OutputType:
        """Parses the output of the LLM's `.invoke` method."""
        raise NotImplementedError

    async def _aparse_output(self, state: dict[str, Any]) -> OutputType:
        """Parses the output of the LLM's `.ainvoke` method. Defaults to calling `_parse_output`."""
        return self._parse_final_output(state)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
        retry=retry_if_exception_type((TimeoutError,)),
    )
    def label(self, input: InputType) -> OutputType:
        """
        Labels the input. If the prepare intermediate method throws a StopMultistepProcessing exception, the
        labeling process is stopped early.
        """
        try:
            prepared_input = self._prepare_initial_input(input)

            for step_no in range(1, self.num_steps + 1):
                try:
                    base_message = self.llms[step_no - 1].invoke(prepared_input)  # type: ignore
                    prepared_input = self._prepare_intermediate(step_no, base_message, prepared_input)
                except StopMultistepProcessing:
                    return self._parse_final_output(prepared_input)

            return self._parse_final_output(prepared_input)
        except Exception as e:
            if self.on_error == "raise":
                raise e
            else:
                logging.warning(f"Error labeling input {input}: {e}")
                return self.error_value

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
        retry=retry_if_exception_type((TimeoutError,)),
    )
    async def alabel(self, input: InputType) -> OutputType:
        """
        Labels the input asynchronously. If the prepare intermediate method throws a StopMultistepProcessing exception,
        the labeling process is stopped early.
        """
        try:
            async with asyncio.timeout(self.timeout_secs):
                prepared_input = self._prepare_initial_input(input)

                for step_no in range(1, self.num_steps + 1):
                    try:
                        base_message = await self.llms[step_no - 1].ainvoke(prepared_input)  # type: ignore
                        prepared_input = self._prepare_intermediate(step_no, base_message, prepared_input)
                    except StopMultistepProcessing:
                        return await self._aparse_output(prepared_input)

                return await self._aparse_output(prepared_input)
        except Exception as e:
            if self.on_error == "raise":
                raise e
            else:
                logging.warning(f"Error labeling input {input}: {e}")
                return self.error_value

    def batch_label(self, inputs: list[InputType]) -> list[OutputType | None]:
        """Labels a batch of inputs."""
        return [self.label(input) for input in inputs]

    async def abatch_label(self, inputs: list[InputType], num_parallel: int = 16) -> list[OutputType | None]:
        """Labels a batch of inputs asynchronously."""

        async def _do_label(input: InputType, sem: asyncio.Semaphore) -> OutputType | None:
            async with sem:
                try:
                    return await self.alabel(input)
                except Exception as e:  # noqa catch-all errors for now
                    logging.exception(f"Error labeling input {input}: {e}")

                    if self.on_error == "raise":
                        raise e
                    else:
                        logging.warning(f"Error labeling input {input}: {e}")
                        return self.error_value

        sem = asyncio.Semaphore(num_parallel)

        return await tqdm_asyncio.gather(*[_do_label(input, sem) for input in inputs])  # type: ignore


class WildChatRealismLabeler(LLMLabeler[str, bool]):
    """
    Represents an LLM annotator for determining whether a string is similar to user prompts in the WildChat dataset. The
    algorithm works as follows:
    - Fetch the top-21 most similar embeddings from n (default of 5000) examples from WildChat
    - Ask the LLM if the top-1 most similar prompt is more similar to the top-2->21 (20 total) than the given prompt
    - Return true if it is; false otherwise
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embedding_model: Embeddings,
        num_dataset_examples: int = 5000,
        num_discard_initial_examples: int = 50000,
        country_filter: str | None = "United States",
    ) -> None:
        super().__init__(llm)
        self.dataset: Dataset = load_dataset("allenai/WildChat-1M")
        self.wildchat_prompts: list[str] = []

        # Sample prompts from the dataset
        ds_split = self.dataset["train"]
        row_idx = 0

        while len(self.wildchat_prompts) < num_dataset_examples and row_idx < len(ds_split):
            if row_idx < num_discard_initial_examples:
                row_idx += 1
                continue

            row = ds_split[row_idx]
            row_idx += 1

            if country_filter is None or row["country"] == country_filter:
                prompt = row["conversation"][0]["content"]
                self.wildchat_prompts.append(prompt)

        # Prepare the vector database
        self.wildchat_documents: list[Document] = [Document(x) for x in self.wildchat_prompts]
        self.db: FAISS = cached_get_database(self.wildchat_documents, embedding_model)

    @property
    def error_value(self) -> bool:
        return False

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return WILDCHAT_REALISM_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: str) -> dict[str, Any]:
        wildchat_prompts = [d.page_content for d in self.db.similarity_search(input, k=21)]
        nearest_prompt = wildchat_prompts[0]
        all_prompts = [f"{x[:100]}..." for x in wildchat_prompts[1:]]
        wildchat_prompt_str = "\n - ".join(all_prompts)

        if random.random() < 0.5:
            prompt1 = nearest_prompt
            prompt2 = input
            self.asyncio_context.set(dict(reversed=True))
        else:
            prompt1 = input
            prompt2 = nearest_prompt
            self.asyncio_context.set(dict(reversed=False))

        return dict(prompt1=prompt1, prompt2=prompt2, wildchat_prompt_str=wildchat_prompt_str)

    def _parse_output(self, output: BaseMessage) -> bool:
        assert output.content is not None and isinstance(output.content, str)
        content = output.content.strip()

        return content.strip() == "2" if self.asyncio_context.get().get("reversed") else content.strip() == "1"


class SummarizingQuicktakeLabeler(MultistepLLMLabeler[str, str]):
    """
    Represents an LLM annotator for producing quicktake summaries.
    """

    @property
    def error_value(self) -> str:
        return ""

    @property
    def num_steps(self) -> int:
        return 2

    def _prepare_initial_input(self, input: str) -> dict[str, Any]:
        return dict(prompt=input)

    def _prepare_intermediate(self, step_no: int, base_message: BaseMessage, state: dict[str, Any]) -> dict[str, Any]:
        match step_no:
            case 1:
                state["long_response"] = str(base_message.content).strip()
            case 2:
                state["short_response"] = str(base_message.content).strip()
            case _:
                raise ValueError(f"Invalid step number: {step_no}")

        return state

    def _parse_final_output(self, state: dict[str, Any]) -> str:
        return str(state["short_response"])

    def _prepare_llm(self, step_no: int, llm: BaseChatModel) -> BaseChatModel:
        match step_no:
            case 1:
                return QUICKTAKE_SUMMARIZING_PROMPT_TEMPLATE_1 | llm  # type: ignore
            case 2:
                return QUICKTAKE_SUMMARIZING_PROMPT_TEMPLATE_2 | llm  # type: ignore
            case _:
                raise ValueError(f"Invalid step number: {step_no}")
