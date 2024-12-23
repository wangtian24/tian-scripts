import json
import re
from typing import Any

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

from ypl.backend.llm.labeler import EmbeddingsLabeler, MultistepLLMLabeler
from ypl.backend.prompts import SEMANTIC_DIFF_PROMPT_TEMPLATE
from ypl.utils import simple_strip, split_markdown_list


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
        spans1 = [span for span in spans1 if span.level >= threshold]
        spans2 = [span for span in spans2 if span.level >= threshold]

        if len(spans1) > 0.7 * len(text1.split("\n")) or len(spans2) > 0.7 * len(text2.split("\n")):
            return ([], [])

        return spans1, spans2

    def _highlight(self, text1: str, text2: str) -> tuple[list[HighlightSpan], list[HighlightSpan]]:
        raise NotImplementedError


def visualize_highlight(text: str, spans: list[HighlightSpan], threshold: float = 0.5) -> str:
    strings = []
    last_end = 0

    # Print with yellow color
    for span in spans:
        if span.level <= threshold:
            continue

        strings.append(text[last_end : span.start])
        strings.append(f"\033[93m{text[span.start:span.end]}\033[0m")
        last_end = span.end

    strings.append(text[last_end:])

    return "".join(strings)


def _should_label(
    text1: str,
    text2: str,
    md_list1: list[tuple[int, int]],
    md_list2: list[tuple[int, int]],
    scale_threshold: float = 2.5,
    max_length: int = 500,
    max_std: float = 40,
    min_proportion: float = 0.75,
) -> bool:
    """
    Returns True if the two texts should be labeled. Currently, they are only labeled if they have
    - Similar mean line length
    - Similar standard deviation of line length
    - Lines aren't too different in length (max_std)
    - Any line isn't too long (max_length)
    - The two texts have a similar number of lines

    Args:
        text1: The first text.
        text2: The second text.
        md_list1: The list of markdown lists for the first text.
        md_list2: The list of markdown lists for the second text.
        scale_threshold: The threshold for the ratio of the mean line lengths.
        max_length: The maximum length of a line.
        max_std: The maximum standard deviation of line lengths.
        min_proportion: The minimum proportion of the number of lines.
    """
    if not md_list1 or not md_list2:
        return False

    md_len_arr1 = np.array([end - start for start, end in md_list1])
    md_len_arr2 = np.array([end - start for start, end in md_list2])
    len_mean1, len_std1 = np.mean(md_len_arr1), np.std(md_len_arr1)
    len_mean2, len_std2 = np.mean(md_len_arr2), np.std(md_len_arr2)

    len_readable_chars1 = len(re.sub(r"\s", "", text1))
    len_readable_chars2 = len(re.sub(r"\s", "", text2))

    if len_mean1 * len(md_len_arr1) < min_proportion * len_readable_chars1:
        return False

    if len_mean2 * len(md_len_arr2) < min_proportion * len_readable_chars2:
        return False

    if (
        len_mean1 / len_mean2 > scale_threshold
        or len_mean2 / len_mean1 > scale_threshold
        or max(md_len_arr1) > max_length
        or max(md_len_arr2) > max_length
        or max(len_std1, len_std2) > max_std
    ):
        return False

    return True


class EmbeddingsSemanticDifferenceHighlighter(
    SemanticDifferenceHighlighter,
    EmbeddingsLabeler[tuple[str, str], tuple[list[HighlightSpan], list[HighlightSpan]]],
):
    """
    Takes two texts as a tuple and highlights the semantic differences between them.
    """

    def _highlight(self, text1: str, text2: str) -> tuple[list[HighlightSpan], list[HighlightSpan]]:
        md_list1 = list(split_markdown_list(text1))
        md_list2 = list(split_markdown_list(text2))

        if not _should_label(text1, text2, md_list1, md_list2):
            return ([], [])  # only label if they seem "similar" enough

        spans1, spans2 = self.label(input=(text1, text2))

        return spans1, spans2

    def _prepare_input(self, input: tuple[str, str]) -> tuple[dict[str, Any], list[str]]:
        text1, text2 = input
        md_list1 = list(split_markdown_list(text1))
        md_list2 = list(split_markdown_list(text2))

        strings1 = [text1[start:end] for start, end in md_list1]
        strings2 = [text2[start:end] for start, end in md_list2]

        return dict(
            strings1=strings1,
            strings2=strings2,
            md_list1=md_list1,
            md_list2=md_list2,
        ), ["strings1", "strings2"]

    def _parse_output(
        self, embeddings: dict[str, list[list[float]]], state: dict[str, Any]
    ) -> tuple[list[HighlightSpan], list[HighlightSpan]]:
        X1 = np.array(embeddings["strings1"])
        X2 = np.array(embeddings["strings2"])
        X1 = np.transpose(X1)
        cosine_sim = np.dot(X2, X1)  # embeddings must already normalized
        most_sim1 = np.max(cosine_sim, axis=0)
        most_sim2 = np.max(cosine_sim, axis=1)

        # Add hybrid tf-idf to the mix
        tf_idf = TfidfVectorizer()
        strings1 = state["strings1"]
        strings2 = state["strings2"]
        tf_idf.fit(strings1 + strings2)
        tf_idf_vecs1 = tf_idf.transform([simple_strip(x) for x in strings1])
        tf_idf_vecs2 = tf_idf.transform([simple_strip(x) for x in strings2])
        tf_idf_sim = np.dot(tf_idf_vecs2, tf_idf_vecs1.T)
        tf_idf_sim1 = np.array(np.max(tf_idf_sim, axis=0).todense()).reshape(-1)
        tf_idf_sim2 = np.array(np.max(tf_idf_sim, axis=1).todense()).reshape(-1)
        a = 0.75

        return [
            HighlightSpan(start=start, end=end, level=1 - (a * level + (1 - a) * tf_idf_level))
            for (start, end), level, tf_idf_level in zip(state["md_list1"], most_sim1, tf_idf_sim1, strict=False)
        ], [
            HighlightSpan(start=start, end=end, level=1 - (a * level + (1 - a) * tf_idf_level))
            for (start, end), level, tf_idf_level in zip(state["md_list2"], most_sim2, tf_idf_sim2, strict=False)
        ]


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
        md_list1 = list(split_markdown_list(text1))
        md_list2 = list(split_markdown_list(text2))

        if not _should_label(text1, text2, md_list1, md_list2):
            return ([], [])  # only label if they seem "similar" enough

        spans1, spans2 = self.label(input=(text1, text2))

        return spans1, spans2

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
            level = level / 100
            spans.append(HighlightSpan(start=start, end=end, level=level))

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
