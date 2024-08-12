from dataclasses import dataclass
from typing import Any


def combine_short_sentences(
    sentences: list[str], max_combined_length: int = 40, max_single_length: int = 10
) -> list[str]:
    """Combine short sentences into a single sentence."""
    combined_sentences = []
    current_sentence = ""

    for sentence in sentences:
        if (len(current_sentence) + len(sentence) <= max_combined_length) or (
            len(current_sentence) < max_single_length
        ):
            current_sentence += " " + sentence if current_sentence else sentence
        else:
            if current_sentence:
                combined_sentences.append(current_sentence.strip())
            current_sentence = sentence

    if current_sentence:
        combined_sentences.append(current_sentence.strip())

    return combined_sentences


@dataclass
class Battle:
    model_a: str
    model_b: str
    # Convention is between [0..1], where 0 means "loss" and 1 means "win".
    result_a: float

    def winner(self) -> str | None:
        return self.model_a if self.result_a > 0.5 else self.model_b if self.result_a < 0.5 else None

    def loser(self) -> str | None:
        return self.model_b if self.result_a > 0.5 else self.model_a if self.result_a < 0.5 else None

    def tie(self) -> bool:
        return self.result_a == 0.5


@dataclass
class AnnotatedFloat:
    """An annotated value."""

    value: float | None
    annotation: str | None

    def __float__(self) -> float | None:
        return self.value


@dataclass
class RankedModel:
    """A model with a rank."""

    model: str
    rank: AnnotatedFloat

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "rank": self.rank.value, "annotation": self.rank.annotation}
