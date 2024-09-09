import asyncio
import logging
from contextvars import ContextVar
from typing import Any, Generic, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from tqdm.asyncio import tqdm_asyncio

from backend.prompts import JUDGE_YUPP_CHAT_PROMPT

logging.basicConfig(level=logging.INFO)
InputType = TypeVar("InputType")
JudgementType = TypeVar("JudgementType")


class LLMJudge(Generic[InputType, JudgementType]):
    """
    Represents an LLM that takes in objects of type `InputType` and outputs a judgement of type `JudgementType`.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = self._prepare_llm(llm)
        self.asyncio_context: ContextVar = ContextVar("Coroutine local")

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        """Prepares the LLM for judging. The default implementation is a no-op."""
        return llm

    def _prepare_input(self, input: InputType) -> dict[str, Any]:
        """Prepares the input to pass into the LLM's `.invoke` method."""
        raise NotImplementedError

    def _parse_output(self, output: BaseMessage) -> JudgementType:
        """Parses the output of the LLM's `.invoke` method."""
        raise NotImplementedError

    async def _aparse_output(self, output: BaseMessage) -> JudgementType:
        """Parses the output of the LLM's `.ainvoke` method. Defaults to calling `_parse_output`."""
        return self._parse_output(output)

    def judge(self, input: InputType) -> JudgementType:
        """Judges the input."""
        prepared_input = self._prepare_input(input)
        output = self.llm.invoke(prepared_input)  # type: ignore

        return self._parse_output(output)

    async def ajudge(self, input: InputType) -> JudgementType:
        """Judges the input asynchronously."""
        prepared_input = self._prepare_input(input)
        output = await self.llm.ainvoke(prepared_input)  # type: ignore

        return await self._aparse_output(output)

    def batch_judge(self, inputs: list[InputType]) -> list[JudgementType | None]:
        """Judges a batch of inputs."""
        return [self.judge(input) for input in inputs]

    async def abatch_judge(self, inputs: list[InputType], num_parallel: int = 16) -> list[JudgementType | None]:
        """Judges a batch of inputs asynchronously."""

        async def _do_judge(input: InputType, sem: asyncio.Semaphore) -> JudgementType | None:
            async with sem:
                try:
                    return await self.ajudge(input)
                except Exception as e:  # noqa catch-all errors for now
                    logging.exception(f"Error judging input {input}: {e}")
                    return None

        sem = asyncio.Semaphore(num_parallel)

        return await tqdm_asyncio.gather(*[_do_judge(input, sem) for input in inputs])  # type: ignore


class YuppEvaluationJudge(LLMJudge[tuple[str, str, str], int]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_YUPP_CHAT_PROMPT | llm  # type: ignore

    def _prepare_input(self, input: tuple[str, str, str]) -> dict[str, Any]:
        return dict(response1=input[1], response2=input[2], user_prompt=input[0])

    def _parse_output(self, output: BaseMessage) -> int:
        try:
            # mypy doesn't like this, even though content is type-annotated as a string union
            return int(output.content.strip()[0])  # type: ignore
        except (ValueError, IndexError) as e:
            logging.exception(f"Error parsing output {output}: {e}")
            return -1  # return -1 if we can't parse the output
