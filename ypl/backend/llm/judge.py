import json
import random
import re
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic.v1 import BaseModel as BaseModelV1

from ypl.backend.llm.chat import ModelInfo
from ypl.backend.llm.constants import MODEL_HEURISTICS
from ypl.backend.llm.labeler import InputType, LLMLabeler, OnErrorBehavior, OutputType
from ypl.backend.prompts import (
    FEEDBACK_QUALITY_PROMPT_TEMPLATE,
    JUDGE_RESPONSE_REFUSAL_PROMPT,
    JUDGE_YUPP_CHAT_PROMPT_SPEED_AWARE_TEMPLATE,
    JUDGE_YUPP_CHAT_PROMPT_TEMPLATE,
    JUDGE_YUPP_ONLINE_PROMPT_TEMPLATE,
    JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_SIMPLE_TEMPLATE,
    JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_TEMPLATE,
    PROMPT_MULTILABEL_CLASSIFICATION_PROMPT_TEMPLATE,
    RESPONSE_DIFFICULTY_PROMPT_TEMPLATE,
    RESPONSE_QUALITY_PROMPT_TEMPLATE,
)

DEFAULT_PROMPT_DIFFICULTY = 4  # Most common value.
LOW_PROMPT_DIFFICULTY = 1


class JudgeConfig(BaseModelV1):
    llms: list[ModelInfo] = []
    choice_strategy: Literal["random", "min_cost"] = "min_cost"
    timeout: int = 5  # seconds


def choose_llm(
    models: list[ModelInfo],
    strategy: Literal["random", "min_cost"] = "min_cost",
    exclude_models: set[str] | None = None,
    seed: int | None = None,
) -> ModelInfo:
    seed = random.randint(0, 2**31) if seed is None else seed
    random.seed(seed)
    exclude_models = exclude_models or set()
    models = list([model for model in models if model.model not in exclude_models])

    assert models, "No models to choose from"

    match strategy:
        case "random":
            return random.choice(models)
        case "min_cost":
            # choose the LLM with the lowest cost
            return min(
                models, key=lambda model_info: MODEL_HEURISTICS[model_info.model].dollars_per_million_output_tokens
            )
        case _:
            raise ValueError(f"Invalid choice strategy: {strategy}")


class YuppEvaluationLabeler(LLMLabeler[tuple[str, str, str], int]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_YUPP_CHAT_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: tuple[str, str, str]) -> dict[str, Any]:
        """Tuple is (user_prompt, response1, response2)"""
        return dict(response1=input[1], response2=input[2], user_prompt=input[0])

    def _parse_output(self, output: BaseMessage) -> int:
        return int(str(output.content).strip()[0])

    @property
    def error_value(self) -> int:
        return -1


class SpeedAwareYuppEvaluationLabeler(LLMLabeler[tuple[str, str, str, float, float], int]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_YUPP_CHAT_PROMPT_SPEED_AWARE_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: tuple[str, str, str, float, float]) -> dict[str, Any]:
        """Tuple is (user_prompt, response1, response2, time1, time2)"""
        return dict(response1=input[1], response2=input[2], user_prompt=input[0], time1=input[3], time2=input[4])

    def _parse_output(self, output: BaseMessage) -> int:
        return int(str(output.content).strip()[0])

    @property
    def error_value(self) -> int:
        return -1


class YuppPromptDifficultyLabeler(LLMLabeler[tuple[str, str, str], int]):
    def __init__(
        self,
        llm: BaseChatModel,
        timeout_secs: float = 5.0,
        on_error: OnErrorBehavior = "raise",
        max_words_low_quality: int = 4,  # prompts under this word count are considered low quality.
    ) -> None:
        super().__init__(llm, timeout_secs, on_error)
        self.max_words_low_quality = max_words_low_quality

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: tuple[str, str, str]) -> dict[str, Any]:
        return dict(response1=input[1], response2=input[2], user_prompt=input[0])

    def _parse_output(self, output: BaseMessage) -> int:
        return int(re.search(r"\"overall\":\s*(\d+)", str(output.content)).group(1))  # type: ignore

    @property
    def error_value(self) -> int:
        return -1

    def _heuristic_label(self, input: tuple[str, str, str]) -> tuple[int | None, str]:
        """Labels based on heuristics if possible, otherwise None."""
        num_words = len(input[0].split())
        if num_words <= self.max_words_low_quality:
            return LOW_PROMPT_DIFFICULTY, f"Short prompt ({num_words} <= {self.max_words_low_quality} words)"

        return None, ""

    def label(self, input: InputType) -> OutputType:  # type: ignore
        heuristic_label, _ = self._heuristic_label(input)  # type: ignore
        return heuristic_label or super().label(input)  # type: ignore

    async def alabel(self, input: InputType) -> OutputType:
        heuristic_label, _ = self._heuristic_label(input)  # type: ignore
        return heuristic_label or await super().alabel(input)  # type: ignore

    def label_full(self, input: InputType) -> tuple[OutputType, str]:
        heuristic_label, heuristic_reason = self._heuristic_label(input)  # type: ignore
        return (heuristic_label, heuristic_reason) if heuristic_label else super().label_full(input)  # type: ignore

    async def alabel_full(self, input: InputType) -> tuple[OutputType, str]:
        heuristic_label, heuristic_reason = self._heuristic_label(input)  # type: ignore
        return (heuristic_label, heuristic_reason) if heuristic_label else await super().alabel_full(input)  # type: ignore

    def _clean_output(self, output: BaseMessage) -> str:
        return super()._clean_output(output).replace("json\n", "").replace("```", "")


def label_prompt_difficulty(prompt: str, llm: BaseChatModel) -> int:
    return YuppPromptDifficultyLabeler(llm).label((prompt, "", ""))


class YuppPromptDifficultyLabelerSimple(YuppPromptDifficultyLabeler):
    max_length = 300

    def _prepare_input(self, input: tuple[str, str, str]) -> dict[str, Any]:
        short_input = [txt[: self.max_length] + "..." if len(txt) > self.max_length else txt for txt in input]
        return dict(response1=short_input[1], response2=short_input[2], user_prompt=input[0])

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_SIMPLE_TEMPLATE | llm  # type: ignore


class YuppQualityLabeler(LLMLabeler[tuple[str, str], int]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return RESPONSE_QUALITY_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: tuple[str, str]) -> dict[str, Any]:
        """`input` is a (prompt, response) tuple"""
        return dict(prompt=input[0], response=input[1])

    def _parse_output(self, output: BaseMessage) -> int:
        content = str(output.content)

        if m := re.search(r"\{.*?\"score\":\s*(\d+)\}.*", content):
            return int(m.group(1))

        return self.error_value

    @property
    def error_value(self) -> int:
        return -1


class YuppSingleDifficultyLabeler(LLMLabeler[str, int]):
    """Labels the difficulty of a single prompt without providing a reference, unlike YuppPromptDifficultyLabeler."""

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return RESPONSE_DIFFICULTY_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: str) -> dict[str, Any]:
        return dict(prompt=input)

    def _parse_output(self, output: BaseMessage) -> int:
        content = str(output.content)

        if m := re.search(r"\{.*?\"score\":\s*(\d+)\}.*", content):
            return int(m.group(1))

        return self.error_value

    @property
    def error_value(self) -> int:
        return -1


class YuppOnlinePromptLabeler(LLMLabeler[str, bool]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_YUPP_ONLINE_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: str) -> dict[str, Any]:
        return dict(prompt=input)

    def _parse_output(self, output: BaseMessage) -> bool:
        return "true" in str(output.content)

    @property
    def error_value(self) -> bool:
        return False


class YuppMultilabelClassifier(LLMLabeler[str, list[str]]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return PROMPT_MULTILABEL_CLASSIFICATION_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: str) -> dict[str, Any]:
        return dict(prompt=input)

    def _parse_output(self, output: BaseMessage) -> list[str]:
        content = str(output.content)

        if m := re.search(r"\{.*?\"categories\":\s*(\[.*?\])\}.*", content):
            item = json.loads(m.group(1))

            if not isinstance(item, list):
                return self.error_value

            return item

        return self.error_value

    @property
    def error_value(self) -> list[str]:
        return []


class ResponseRefusalLabeler(LLMLabeler[tuple[str, str], int]):
    def __init__(
        self,
        llm: BaseChatModel,
        timeout_secs: float = 5.0,
        on_error: OnErrorBehavior = "use_error_value",
        max_prompt_len: int = 200,
        max_response_len: int = 300,
    ) -> None:
        super().__init__(llm, timeout_secs, on_error)
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_RESPONSE_REFUSAL_PROMPT | llm  # type: ignore

    def _prepare_input(self, input: tuple[str, str]) -> dict[str, Any]:
        prompt = input[0][: self.max_prompt_len] + "..." if len(input[0]) > self.max_prompt_len else input[0]
        response = input[1][: self.max_response_len] + "..." if len(input[1]) > self.max_response_len else input[1]
        return dict(prompt=prompt, response=response)

    def _parse_output(self, output: BaseMessage) -> int:
        content = str(output.content)
        try:
            return int(content)
        except ValueError:
            return self.error_value

    @property
    def error_value(self) -> int:
        return -1


class FeedbackQualityLabeler(LLMLabeler[str, int]):
    """Labels the quality of feedback on a scale of 1-10."""

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return FEEDBACK_QUALITY_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: str) -> dict[str, Any]:
        return dict(feedback=input)

    def _parse_output(self, output: BaseMessage) -> int:
        content = str(output.content)

        if m := re.search(r"\{.*?\"score\":\s*(\d+)\}.*", content):
            return int(m.group(1))

        return self.error_value

    @property
    def error_value(self) -> int:
        return -1

    def _heuristic_label(self, input: str) -> int | None:
        """Labels based on heuristics if possible, otherwise None."""
        # Very short feedback is likely low quality
        if len(input.split()) <= 4:
            return 1
        return None

    def label(self, input: str) -> int:
        return self._heuristic_label(input) or super().label(input)

    async def alabel(self, input: str) -> int:
        return self._heuristic_label(input) or await super().alabel(input)
