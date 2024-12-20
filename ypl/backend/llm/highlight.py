import json
import re
from collections.abc import Generator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from ypl.backend.llm.labeler import MultistepLLMLabeler
from ypl.backend.prompts import SEMANTIC_DIFF_PROMPT_TEMPLATE


class HighlightSpan(BaseModel):
    start: int  # start index of the span, inclusive
    end: int  # end index of the span, non-inclusive
    level: float = 1.0  # 0.0 to 1.0, with 1.0 being the most highlighted

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HighlightSpan):
            return False

        return self.start == other.start and self.end == other.end


class SemanticDifferenceHighlighter:
    def highlight(
        self, text1: str, text2: str, threshold: float = 0.5
    ) -> tuple[list[HighlightSpan], list[HighlightSpan]]:
        spans1, spans2 = self._highlight(text1, text2)
        return (
            [span for span in spans1 if span.level >= threshold],
            [span for span in spans2 if span.level >= threshold],
        )

    def _highlight(self, text1: str, text2: str) -> tuple[list[HighlightSpan], list[HighlightSpan]]:
        raise NotImplementedError


def visualize_highlight(text: str, spans: list[HighlightSpan]) -> str:
    strings = []
    last_end = 0

    # Print with yellow color
    for span in spans:
        strings.append(text[last_end : span.start])
        strings.append(f"\033[93m{text[span.start:span.end]}\033[0m")
        last_end = span.end

    strings.append(text[last_end:])

    return "".join(strings)


def split_markdown_list(text: str) -> Generator[tuple[int, int], None, None]:
    for match in re.finditer(r"^\s*(\*\s|-|[0-9]+\.).*(\n|$)", text, re.MULTILINE):
        yield match.start(), match.end()


class LLMSemanticDifferenceHighlighter(
    SemanticDifferenceHighlighter, MultistepLLMLabeler[tuple[str, str], tuple[list[HighlightSpan], list[HighlightSpan]]]
):
    """
    Takes two texts as a tuple and highlights the semantic differences between them.
    """

    @property
    def error_value(self) -> tuple[list[HighlightSpan], list[HighlightSpan]]:
        return ([], [])

    @property
    def num_steps(self) -> int:
        return 1  # currently one; will be extended to multiple steps

    def _highlight(self, text1: str, text2: str) -> tuple[list[HighlightSpan], list[HighlightSpan]]:
        if not split_markdown_list(text1) or not split_markdown_list(text2):
            return ([], [])  # only split if there are markdown lists

        return self.label(input=(text1, text2))

    def _prepare_initial_input(self, input: tuple[str, str]) -> dict[str, Any]:
        def _insert_ids(text: str, prefix: str) -> tuple[str, dict[str, tuple[int, int]]]:
            sentences = []
            spans = list(split_markdown_list(text))
            span_map = {}

            for idx, (start, end) in enumerate(spans):
                sentences.append(f"<{prefix}ID{idx}> {text[start:end]}")
                span_map[f"{prefix}ID{idx}"] = (start, end)

            return "\n".join(sentences), span_map

        md_points1, span_map1 = _insert_ids(input[0], "A-")
        md_points2, span_map2 = _insert_ids(input[1], "B-")

        return dict(
            text1=md_points1,
            text2=md_points2,
            span_map1=span_map1,
            span_map2=span_map2,
        )

    def _prepare_intermediate(self, step_no: int, base_message: BaseMessage, state: dict[str, Any]) -> dict[str, Any]:
        match step_no:
            case 1:
                content = str(base_message.content)
                state["differences"] = json.loads(content)
            case _:
                raise ValueError(f"Invalid step number: {step_no}")

        return state

    def _parse_final_output(self, state: dict[str, Any]) -> tuple[list[HighlightSpan], list[HighlightSpan]]:
        spans1: list[HighlightSpan] = []
        spans2: list[HighlightSpan] = []

        for span_id, level in state["differences"]:
            if span_id.startswith("A-"):
                span_map = state["span_map1"]
                spans = spans1
            elif span_id.startswith("B-"):
                span_map = state["span_map2"]
                spans = spans2
            else:
                continue

            if span_id not in span_map:
                continue

            start, end = span_map[span_id]
            spans.append(HighlightSpan(start=start, end=end, level=1 - level))

        spans1.sort(key=lambda x: x.start)
        spans2.sort(key=lambda x: x.start)

        spans1 = list(dict.fromkeys(spans1))
        spans2 = list(dict.fromkeys(spans2))

        return spans1, spans2

    def _prepare_llm(self, step_no: int, llm: BaseChatModel) -> BaseChatModel:
        match step_no:
            case 1:
                return SEMANTIC_DIFF_PROMPT_TEMPLATE | llm  # type: ignore
            case _:
                raise ValueError(f"Invalid step number: {step_no}")
