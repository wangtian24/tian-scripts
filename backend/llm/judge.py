import asyncio
import logging
import random
from contextvars import ContextVar
from typing import Any, Generic, TypeVar

from datasets import Dataset, load_dataset
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from tqdm.asyncio import tqdm_asyncio

from backend.llm.embedding import cached_get_database
from backend.prompts import JUDGE_YUPP_CHAT_PROMPT, WILDCHAT_REALISM_PROMPT_TEMPLATE

logging.basicConfig(level=logging.WARNING)
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


class WildChatRealismJudge(LLMJudge[str, bool]):
    """
    Represents an LLM judge for determining whether a string is similar to user prompts in the WildChat dataset. The
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
