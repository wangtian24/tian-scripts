import logging
import random
import re
from typing import Any, Literal, cast

import orjson
import vertexai
import vertexai.preview
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel as BaseModelV1
from sqlmodel import Session, select
from vertexai.preview.generative_models import GenerativeModel

from ypl.backend.db import get_engine
from ypl.backend.llm.constants import MODEL_HEURISTICS
from ypl.backend.llm.db_helpers import get_yapp_descriptions
from ypl.backend.llm.labeler import InputType, LLMLabeler, OnErrorBehavior, OutputType
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.prompt_classifiers import CategorizerResponse, PromptCategorizer
from ypl.backend.prompts.feedback_quality import (
    FEEDBACK_QUALITY_PROMPT_TEMPLATE,
)
from ypl.backend.prompts.memory_extraction import (
    MEMORY_COMPACTION_PROMPT,
    PROMPT_MEMORY_EXTRACTION_PROMPT_TEMPLATE,
)
from ypl.backend.prompts.prompt_classification import (
    JUDGE_YUPP_ONLINE_PROMPT,
    PROMPT_MULTILABEL_CLASSIFICATION_PROMPT_TEMPLATE,
    get_yapp_classification_prompt_template,
)
from ypl.backend.prompts.prompt_difficulty import (
    JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_SIMPLE_TEMPLATE,
    JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_TEMPLATE,
    JUDGE_YUPP_PROMPT_DIFFICULTY_WITH_COMMENT_PROMPT_TEMPLATE,
)
from ypl.backend.prompts.prompt_modifiers import (
    JUDGE_PROMPT_MODIFIER_PROMPT,
)
from ypl.backend.prompts.quicktake import (
    JUDGE_QUICK_RESPONSE_QUALITY_PROMPT_TEMPLATE,
)
from ypl.backend.prompts.response_quality import (
    JUDGE_RESPONSE_REFUSAL_PROMPT,
    RESPONSE_DIFFICULTY_PROMPT_TEMPLATE,
    RESPONSE_QUALITY_PROMPT_TEMPLATE,
)
from ypl.backend.prompts.review_classification import (
    REVIEW_ROUTE_CLASSIFIER_PROMPT_TEMPLATE,
)
from ypl.backend.prompts.suggestions import (
    JUDGE_CHAT_TITLE_PROMPT_TEMPLATE,
    JUDGE_CONVERSATION_STARTERS_PROMPT_TEMPLATE,
    JUDGE_SUGGESTED_FOLLOWUPS_PROMPT_TEMPLATE,
    JUDGE_SUGGESTED_PROMPTBOX_PROMPT_TEMPLATE,
)
from ypl.backend.prompts.system_prompts import fill_cur_datetime
from ypl.db.chats import PromptModifier

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


class YuppPromptDifficultyLabeler(LLMLabeler[tuple[str, str, str], int]):
    def __init__(
        self,
        llm: BaseChatModel,
        timeout_secs: float = 5.0,
        on_error: OnErrorBehavior = "raise",
        max_words_low_quality: int = 3,  # prompts under this word count are considered low quality.
        max_length: int = 300,
    ) -> None:
        super().__init__(llm, timeout_secs, on_error)
        self.max_words_low_quality = max_words_low_quality
        self.max_length = max_length

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: tuple[str, str, str]) -> dict[str, Any]:
        short_input = [txt[: self.max_length] + "..." if len(txt) > self.max_length else txt for txt in input]
        return dict(response1=short_input[1], response2=short_input[2], user_prompt=input[0])

    def _parse_output(self, output: BaseMessage) -> int:
        return int(re.search(r"\"overall\":\s*(\d+)", str(output.content)).group(1))  # type: ignore

    @property
    def error_value(self) -> int:
        return -1

    def _heuristic_label(self, input: tuple[str, str, str]) -> tuple[int | None, str]:
        """Labels based on heuristics if possible, otherwise None."""
        num_words = len(input[0].split())
        if num_words < self.max_words_low_quality:
            details = {"heuristics": f"Short prompt ({num_words} <= {self.max_words_low_quality} words)"}
            return LOW_PROMPT_DIFFICULTY, orjson.dumps(details).decode("utf-8")

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
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_SIMPLE_TEMPLATE | llm  # type: ignore


class YuppPromptDifficultyWithCommentLabeler(YuppPromptDifficultyLabeler):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_YUPP_PROMPT_DIFFICULTY_WITH_COMMENT_PROMPT_TEMPLATE | llm  # type: ignore


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


class TruncatedPromptLabeler(LLMLabeler[str, OutputType]):
    """Truncates the prompt to a maximum length."""

    def __init__(
        self,
        llm: BaseChatModel,
        timeout_secs: float = 5.0,
        on_error: OnErrorBehavior = "use_error_value",
        max_prompt_len: int = 300,
    ) -> None:
        super().__init__(llm, timeout_secs, on_error)
        self.max_prompt_len = max_prompt_len

    def _prepare_input(self, input: str) -> dict[str, Any]:
        return dict(prompt=input[: self.max_prompt_len] + "..." if len(input) > self.max_prompt_len else input)


class YuppOnlinePromptLabeler(PromptCategorizer, TruncatedPromptLabeler[bool]):
    cached = True

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        cp_template = ChatPromptTemplate.from_messages([("human", fill_cur_datetime(JUDGE_YUPP_ONLINE_PROMPT))])
        return cp_template | llm  # type: ignore

    def _parse_output(self, output: BaseMessage) -> bool:
        return "true" in str(output.content).lower()

    def categorize(self, user_prompt: str) -> CategorizerResponse:
        return CategorizerResponse(category="online" if self.label(user_prompt) else "offline")

    @property
    def error_value(self) -> bool:
        return False


class FastVertexAIOnlinePromptLabeler(PromptCategorizer):
    def __init__(self, project_id: str, region: str, model: str = "gemini-1.5-flash-002") -> None:
        self.project_id = project_id
        self.region = region
        self._init = False

        try:
            vertexai.init(project=self.project_id, location=self.region)
            self.model = GenerativeModel(model)
            self._init = True
        except Exception as e:
            logging.error(f"Error initializing Vertex AI: {e}")

    def _infer_category(self, user_prompt: str) -> str:
        assert self.model is not None
        response = self.model.generate_content([user_prompt], temperature=0.0, max_tokens=16)
        return "online" if response.text else "offline"

    def categorize(self, user_prompt: str) -> CategorizerResponse:
        if not self._init:
            return CategorizerResponse(category="offline")

        return CategorizerResponse(category=self._infer_category(user_prompt))


class YuppMultilabelClassifier(LLMLabeler[str, list[str]]):
    cached = True

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return PROMPT_MULTILABEL_CLASSIFICATION_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: str) -> dict[str, Any]:
        return dict(prompt=input)

    def _parse_output(self, output: BaseMessage) -> list[str]:
        content = str(output.content)

        if m := re.search(r"\{.*?\"categories\":\s*(\[.*?\])\}.*", content):
            item = orjson.loads(m.group(1))

            if not isinstance(item, list):
                return self.error_value

            return item

        return self.error_value

    @property
    def error_value(self) -> list[str]:
        return []


class YappAgentClassifier(LLMLabeler[str, str]):
    """a simple agent classification model to help pick from available Yapp agents"""

    cached = True

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        yapp_descriptions: dict[str, str] = get_yapp_descriptions()
        return get_yapp_classification_prompt_template(yapp_descriptions) | llm  # type: ignore

    def _prepare_input(self, input: str) -> dict[str, Any]:
        return dict(prompt=input)

    def _parse_output(self, output: BaseMessage) -> str:
        content = str(output.content).lower()
        if content in ["none", "weather-yapp", "news-yapp", "wikipedia-yapp", "youtube-transcript-yapp"]:
            return content

        return self.error_value

    @property
    def error_value(self) -> str:
        return "none"


class YuppMemoryExtractor(LLMLabeler[list[BaseMessage], list[str]]):
    """Accepts messages in a single user and system turn, and finds memories."""

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        cp_template = ChatPromptTemplate.from_messages(
            [("human", fill_cur_datetime(PROMPT_MEMORY_EXTRACTION_PROMPT_TEMPLATE))]
        )
        return cp_template | llm  # type: ignore

    def _prepare_input(self, input: list[BaseMessage]) -> dict[str, str]:
        return dict(chat_history=_format_message_history(input))

    def _parse_output(self, output: BaseMessage) -> list[str]:
        content = str(output.content)

        left, right = content.find("["), content.rfind("]") + 1
        # A parse failure below returns error_value.
        return cast(list[str], orjson.loads(content[left:right]))

    @property
    def error_value(self) -> list[str]:
        return []


class PromptModifierLabeler(TruncatedPromptLabeler[list[str]]):
    modifiers: str = ""

    def init_modifiers(self) -> None:
        with Session(get_engine()) as session:
            results = session.exec(
                select(PromptModifier.name, PromptModifier.text).where(PromptModifier.deleted_at.is_(None))  # type: ignore
            ).all()
            self.modifiers = "\n" + "\n".join([f"{name}: {text}" for name, text in results]) + "\n"

    def _prepare_input(self, input: str) -> dict[str, Any]:
        if not self.modifiers:
            # Get them for the first time.
            self.init_modifiers()
        input_with_prompt = super()._prepare_input(input)
        return input_with_prompt | dict(modifiers=self.modifiers)

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_PROMPT_MODIFIER_PROMPT | llm  # type: ignore

    def _parse_output(self, output: BaseMessage) -> list[str]:
        try:
            return [item.strip() for item in str(output.content).split(",")]
        except ValueError:
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


class QuickResponseQualityLabeler(LLMLabeler[tuple[str, str, list[dict[str, str]]], str]):
    """Labels the quality of quick responses.

    Returns:
        int: Rating from 1-3 where:
        1 - POOR quality (unhelpful, incorrect, or inappropriate)
        2 - ACCEPTABLE quality (meets basic requirements)
        3 - EXCELLENT quality (concise, accurate, and helpful)
    """

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_QUICK_RESPONSE_QUALITY_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: tuple[str, str, list[dict[str, str]]]) -> dict[str, Any]:
        """input is a (prompt, response, chat_history) tuple"""
        chat_history_str = input[2] if input[2] else "No previous conversation"

        return {"user_prompt": input[0], "response": input[1], "chat_history": chat_history_str}

    def _parse_output(self, output: BaseMessage) -> str:
        content = str(output.content).strip()
        if "1" in content:
            return "poor"
        elif "2" in content:
            return "acceptable"
        elif "3" in content:
            return "excellent"
        return self.error_value

    @property
    def error_value(self) -> str:
        return "error"


def _load_json_suggested_prompts(content: str) -> list[dict[str, str]]:
    # Remove optional markdown formatting in the response before parsing.
    content = str(content).replace("```json", "").replace("```", "")
    res = orjson.loads(content)
    if not isinstance(res, list):
        raise ValueError(f"Unexpected output: {content}")
    return res


def _format_message_history(message_history: list[BaseMessage]) -> str:
    chat_history = ""
    for message in message_history:
        if message.type == "human":
            chat_history += f"User: {message.content}\n"
        elif message.type in ["ai", "assistant"]:
            chat_history += f"Assistant: {message.content}\n"
    return chat_history


class SuggestedFollowupsLabeler(LLMLabeler[list[BaseMessage], list[dict[str, str]]]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_SUGGESTED_FOLLOWUPS_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: list[BaseMessage]) -> dict[str, str]:
        return dict(chat_history=_format_message_history(input))

    def _parse_output(self, output: BaseMessage) -> list[dict[str, str]]:
        """Output is a list of suggestions and a labels for them.

        For example, for the input:

            messages = [
                HumanMessage(content="Who won Wimbledon this year?"),
                AIMessage(content="Novak Djokovic won Wimbledon this year."),
            ]

        The output could be:

            [
                {
                    "suggestion": "What was his performance like throughout the tournament?",
                    "label": "Djokovic's Performance"
                },
                {
                    "suggestion": "How does this win compare to his previous Wimbledon victories?",
                    "label": "Historical Comparison"
                },
                {
                    "suggestion": "What are the biggest challenges Djokovic faced this year at Wimbledon?",
                    "label": "Challenges Faced"
                }
            ]
        """
        return _load_json_suggested_prompts(str(output.content))

    @property
    def error_value(self) -> list[dict[str, str]]:
        return []


class SuggestedPromptboxLabeler(LLMLabeler[list[BaseMessage], str]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_SUGGESTED_PROMPTBOX_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: list[BaseMessage]) -> dict[str, str]:
        return dict(chat_history=_format_message_history(input))

    def _parse_output(self, output: BaseMessage) -> str:
        return str(output.content).strip()

    @property
    def error_value(self) -> str:
        return "Ask a follow-up"


class ConversationStartersLabeler(LLMLabeler[list[list[BaseMessage]], list[dict[str, str]]]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_CONVERSATION_STARTERS_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: list[list[BaseMessage]]) -> dict[str, str]:
        full_history = ""
        for chat_history in input:
            full_history += "Conversation Summary:\n\n"
            full_history += _format_message_history(chat_history)
            full_history += "\nEnd of Conversation\n\n"
        return dict(chat_history=full_history)

    def _parse_output(self, output: BaseMessage) -> list[dict[str, str]]:
        """Output is a list of suggestions and a labels for them, in a format similar to SuggestedFollowupsLabeler."""
        return _load_json_suggested_prompts(str(output.content))

    @property
    def error_value(self) -> list[dict[str, str]]:
        return []


class ChatTitleLabeler(LLMLabeler[list[BaseMessage], str]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return JUDGE_CHAT_TITLE_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, input: list[BaseMessage]) -> dict[str, str]:
        return dict(chat_history=_format_message_history(input))

    def _parse_output(self, output: BaseMessage) -> str:
        return str(output.content)

    @property
    def error_value(self) -> str:
        return ""


class MemoryCompactor(LLMLabeler[list[str], list[str]]):
    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return MEMORY_COMPACTION_PROMPT | llm  # type: ignore

    def _prepare_input(self, input: list[str]) -> dict[str, str]:
        return dict(memories="\n".join(input))

    def _parse_output(self, output: BaseMessage) -> list[str]:
        if not output.content:
            return []
        lines = [line.strip() for line in str(output.content).split("\n")]
        return [line for line in lines if line]

    @property
    def error_value(self) -> list[str]:
        return []


class ReviewRouteClassifier(LLMLabeler[str, str]):
    """
    Classifier that determines whether to route a query to Pro Review or Cross-Check.

    - PRO: Factual, math, puzzle, factoid problems that ideally have clear right/wrong answers
    - CROSS: More elaborate queries requiring comparison of style, tone, and subjective factors
    """

    cached = True

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return cast(BaseChatModel, REVIEW_ROUTE_CLASSIFIER_PROMPT_TEMPLATE | llm)

    def _prepare_input(self, input: str) -> dict[str, Any]:
        return {"query": input}

    def _parse_output(self, output: BaseMessage) -> str:
        content = str(output.content).strip().upper()
        if "PRO" in content:
            return "PRO"
        elif "CROSS" in content:
            return "CROSS"
        else:
            # Default to PRO for uncertain cases
            logging.warning(f"Unclear classification result: {content}. Defaulting to PRO.")
            return "PRO"

    @property
    def error_value(self) -> str:
        # Default to PRO if there's an error
        return "PRO"
