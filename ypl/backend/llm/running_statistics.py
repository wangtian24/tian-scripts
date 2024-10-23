import time
from functools import cache

from ypl.backend.llm.model.model import get_model_base_statistics
from ypl.backend.llm.routing.route_data_type import (
    InstantaneousLanguageModelStatistics,
    LanguageModelStatistics,
    StatisticsHistory,
)


class RunningStatisticsTracker:
    """
    Keeps track of the running live statistics, namely latency and throughput, for each model, which may display
    variations due to outages, etc.
    """

    def __init__(self, decay_rate: float = 0.93) -> None:
        """
        decay_rate: The decay rate for the running statistics, defined as the amount to decay the statistics to the
            base statistics every minute.
        update_rate: The rate at which to update the statistics with the latest statistics.
        """
        self.decay_rate = decay_rate
        self.statistics: dict[str, tuple[StatisticsHistory, float]] = {}

    async def _decay_statistics(self, model_id: str) -> None:
        """
        Decay the statistics for a model by the decay rate over the time since the last update, interpolating between
        the base statistics and the current running statistics. If any base statistics are missing, this will be
        a no-op for the missing fields.
        """
        if model_id not in self.statistics:
            return

        last_time = self.statistics[model_id][1]
        time_diff = time.time() - last_time
        rate = self.decay_rate ** (time_diff / 60)
        base_stats_history = StatisticsHistory.from_single(await get_model_base_statistics(model_id))
        curr_stats_history = self.statistics[model_id][0]

        new_stats_history = base_stats_history * (1 - rate) + curr_stats_history * rate
        new_stats_history.replace_missing(curr_stats_history)

        self.statistics[model_id] = (new_stats_history, time.time())

    async def update_statistics(
        self, model_id: str, statistics: InstantaneousLanguageModelStatistics
    ) -> LanguageModelStatistics:
        if model_id not in self.statistics:
            self.statistics[model_id] = (StatisticsHistory.from_single(statistics), time.time())
        else:
            self.statistics[model_id] = (
                self.statistics[model_id][0].append_to_history(statistics),
                time.time(),
            )

        return self.statistics[model_id][0].estimate_statistics()

    async def get_statistics(self, model_id: str) -> LanguageModelStatistics:
        await self._decay_statistics(model_id)

        try:
            return self.statistics[model_id][0].estimate_statistics()
        except KeyError:
            return await get_model_base_statistics(model_id)  # type: ignore[no-any-return]

    @classmethod
    @cache
    def get_instance(cls) -> "RunningStatisticsTracker":
        return cls()
