import random
import re
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic.v1 import BaseModel as BaseModelV1

from ypl.backend.llm.chat import ModelInfo
from ypl.backend.llm.constants import MODEL_HEURISTICS
from ypl.backend.llm.labeler import LLMLabeler
from ypl.backend.prompts import (
    JUDGE_YUPP_CHAT_PROMPT_SPEED_AWARE_TEMPLATE,
    JUDGE_YUPP_CHAT_PROMPT_TEMPLATE,
    JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_TEMPLATE,
    RESPONSE_QUALITY_PROMPT_TEMPLATE,
)


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


class YuppPromptDifficultyLabeler(LLMLabeler[tuple[str, str, str], str]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: tuple[str, str, str]) -> dict[str, Any]:
        return dict(response1=input[1], response2=input[2], user_prompt=input[0])

    def _parse_output(self, output: BaseMessage) -> str:
        return str(output.content).strip()

    @property
    def error_value(self) -> str:
        return "DUNNO"


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
