from collections import deque
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class InstantaneousLanguageModelStatistics(BaseModel):
    first_token_latency_ms: float | None = None
    output_tps: float | None = None


class LanguageModelStatistics(BaseModel):
    first_token_p50_latency_ms: float | None = None
    first_token_p90_latency_ms: float | None = None
    output_p50_tps: float | None = None
    output_p90_tps: float | None = None


class StatisticsHistory(BaseModel):
    first_token_latencies_ms_history: deque[float] = Field(default_factory=deque)
    output_tps_history: deque[float] = Field(default_factory=deque)
    history_size: int = 40

    def model_post_init(self, _: Any) -> None:
        self.trim_history()

    def append_to_history(self, statistics: InstantaneousLanguageModelStatistics) -> "StatisticsHistory":
        self.append_to_latency_history(statistics.first_token_latency_ms)
        self.append_to_tps_history(statistics.output_tps)

        return self

    def append_to_latency_history(self, latency: float | None) -> "StatisticsHistory":
        if latency is not None:
            self.first_token_latencies_ms_history.append(latency)
            self.trim_history()

        return self

    def append_to_tps_history(self, tps: float | None) -> "StatisticsHistory":
        if tps is not None:
            self.output_tps_history.append(tps)
            self.trim_history()

        return self

    def __rmul__(self, other: float) -> "StatisticsHistory":
        return self * other

    def __mul__(self, other: float) -> "StatisticsHistory":
        new_dict = {}

        for k, v in self.model_dump().items():
            match v:
                case deque():
                    new_dict[k] = deque(other * x for x in v)
                case _:
                    new_dict[k] = v

        return StatisticsHistory(**new_dict)  # type: ignore[arg-type]

    def estimate_statistics(self) -> LanguageModelStatistics:
        stats = LanguageModelStatistics()

        if self.first_token_latencies_ms_history:
            stats.first_token_p50_latency_ms = float(np.median(self.first_token_latencies_ms_history))
            stats.first_token_p90_latency_ms = float(np.quantile(self.first_token_latencies_ms_history, 0.9))

        if self.output_tps_history:
            stats.output_p50_tps = float(np.median(self.output_tps_history))
            stats.output_p90_tps = float(np.quantile(self.output_tps_history, 0.9))

        return stats

    def trim_history(self) -> None:
        for q in (self.first_token_latencies_ms_history, self.output_tps_history):
            while len(q) > self.history_size:
                q.popleft()

    def replace_missing(self, other: "StatisticsHistory") -> None:
        """Replaces missing history with the history from the other statistics."""
        if not self.first_token_latencies_ms_history:
            self.first_token_latencies_ms_history = other.first_token_latencies_ms_history

        if not self.output_tps_history:
            self.output_tps_history = other.output_tps_history

    def __add__(self, other: "StatisticsHistory") -> "StatisticsHistory":
        new_dict = {}

        for k, v in self.model_dump().items():
            other_v = getattr(other, k)

            match (v, other_v):
                case (None, None):
                    pass
                case (None, _):
                    new_dict[k] = other_v
                case (_, None):
                    new_dict[k] = v
                case (deque(), deque()):
                    assert (
                        len(v) == len(other_v) or len(v) == 1 or len(other_v) == 1 or len(v) == 0 or len(other_v) == 0
                    ), "Deques must be broadcastable"

                    if len(v) == 1:
                        v = deque(v[0] for _ in other_v)
                    elif len(v) == 0:
                        continue

                    if len(other_v) == 1:
                        other_v = deque(other_v[0] for _ in v)
                    elif len(other_v) == 0:
                        continue

                    new_dict[k] = deque(x + y for x, y in zip(v, other_v, strict=False))
                case _:
                    new_dict[k] = v

        s = StatisticsHistory(**new_dict)
        s.trim_history()

        return s

    @classmethod
    def from_single(
        cls, statistics: LanguageModelStatistics | InstantaneousLanguageModelStatistics
    ) -> "StatisticsHistory":
        s = cls()

        match statistics:
            case LanguageModelStatistics():
                s.append_to_latency_history(statistics.first_token_p50_latency_ms)
                s.append_to_tps_history(statistics.output_p50_tps)
            case InstantaneousLanguageModelStatistics():
                s.append_to_history(statistics)
            case _:
                raise ValueError(f"Invalid statistics type: {type(statistics)}")

        return s


class PreferredModel(BaseModel):
    models: list[str] = Field(description="List of models presented to the user for a given turn.")
    preferred: str | None = Field(description="Which model was preferred by the user, or None if all are bad")
    has_evaluation: bool = True


class RoutingPreference(BaseModel):
    turns: list[PreferredModel] | None = Field(
        description=(
            "The preference for each of the past turns in the chat context "
            "in chronological order (first turn is the oldest). "
            "An empty list indicates that there were no prior turns."
        )
    )
    user_selected_models: list[str] | None = Field(
        description="The models that the user selected before the chat started."
    )
    same_turn_shown_models: list[str] | None = Field(
        description="The models already shown in the current turn (before Show Me More)."
    )
    user_id: str | None = Field(description="The user ID of the user who is being routed.")
    debug_level: int = 1
