import asyncio
import logging
import threading
import time

from google.cloud import monitoring_v3
from ypl.backend.config import settings

"""
Metric counters. All counters are exported to GCP every EXPORT_INTERVAL_SECS seconds.

1. create and keep using, good for static counters.

    from ypl.backend.utils.monitoring import CounterMetric, ValueMetric

    NUM_REQUESTS = mon.CounterMetric("num_requests")
    NUM_REQUESTS.inc()
    NUM_REQUESTS.inc_by(10)

    SOME_LATENCY_MS = mon.ValueMetric("some_latency_ms")
    SOME_LATENCY_MS.record_value(249)

2. create one the fly, good for counter with dynamic names. be careful not to blow up the namespace!

    from ypl.backend.utils.monitoring import metric_inc, metric_inc_by, metric_record

    metric_inc(f"num_chosen_{model_name}")            # auto-create CounterMetric
    metric_inc_by(f"num_chosen_{model_name}", 10)     # auto-create CounterMetric

    metric_record(f"latency_routing_ms", 834)  # auto-create ValueMetric

All counters will be exported to GCP periodically every EXPORT_INTERVAL_SECS seconds.
"""
GCM_CLIENT: monitoring_v3.MetricServiceClient | None = None
GCP_EXPORT_INTERVAL_SECS = 20
GCP_EXPORT_BATCH_SIZE = 200
# for every single value metric, we keep at most 5 data points per second on average.
MAX_VALUES_TO_KEEP_BEFORE_FLUSHING = GCP_EXPORT_BATCH_SIZE * 5


class BaseMetric:
    """Base class for metrics with thread-safe operations."""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()

    def flush_values(self) -> list[tuple[float, int]]:
        """Get and clear the current values, return the as a list of tuples of (timestamp, value)."""
        raise NotImplementedError


class CounterMetric(BaseMetric):
    """
    A single thread-safe metric that stores an integer counter. Only one value is stored and flushed.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._value = 0

    def inc(self) -> None:
        with self._lock:
            self._value += 1

    def inc_by(self, amount: int) -> None:
        with self._lock:
            self._value += amount

    def flush_values(self) -> list[tuple[float, int]]:
        with self._lock:
            # Note that here we just store one counter value with timestamp at the flushing time,
            # which is the GCP export time, it's not accurate.
            return [(time.time(), self._value)]


class ValueMetric(BaseMetric):
    """
    A thread-safe metric that stores individual values with timestamps.
    This class remembers all values written to it with timestamps.
    """

    # TODO(tian): locking might be improved with the read/write lock split.

    def __init__(self, name: str):
        super().__init__(name)
        # List of (timestamp in seconds since epoch, value) tuples
        self._values: list[tuple[float, int]] = []

    def record_value(self, value: int) -> None:
        with self._lock:
            # Note that here we remember every data point.
            self._values.append((time.time(), value))

            # If we have too many values, remove every other one to reduce memory usage
            if len(self._values) > MAX_VALUES_TO_KEEP_BEFORE_FLUSHING:
                self._values = self._values[::2]  # Keep every other value

    def flush_values(self) -> list[tuple[float, int]]:
        with self._lock:
            values = self._values.copy()
            self._values.clear()
            return values


class MetricsRegistry:
    metric: BaseMetric
    series: monitoring_v3.TimeSeries

    def __init__(self, metric: BaseMetric):
        self.metric = metric

    def get_series(self) -> list[monitoring_v3.TimeSeries]:
        """
        Get timeseries for all data points this metric has cached so far. GCP only allows one point per series, argh.
        """
        all_series = []
        for timestamp, value in self.metric.flush_values():
            series = self._create_series()
            series.points = [
                monitoring_v3.Point(
                    {
                        "interval": self._create_interval(timestamp),
                        "value": {"int64_value": value},
                    }
                )
            ]
            all_series.append(series)
        return all_series

    def _create_series(self) -> monitoring_v3.TimeSeries:
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/" + self.metric.name
        # right now it's all the same, but we can change this later
        match settings.ENVIRONMENT:
            case "local":
                series.resource.type = "global"
            case "staging" | "production":
                # TODO(tian): update this to store more instance/replica information in labels if necessary
                series.resource.type = "global"
            case _:
                series.resource.type = "global"
        return series

    def _create_interval(self, now: float) -> monitoring_v3.TimeInterval:
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        return monitoring_v3.TimeInterval({"end_time": {"seconds": seconds, "nanos": nanos}})


class MetricManager:
    """Register and export counters periodically."""

    def __init__(self, export_interval_secs: int = GCP_EXPORT_INTERVAL_SECS):
        self.export_interval_secs: int = export_interval_secs
        self.registry_map: dict[str, MetricsRegistry] = {}
        self._lock: threading.Lock = threading.Lock()
        self._stop_event: asyncio.Event = asyncio.Event()
        self._export_task: asyncio.Task | None = None

    def set_up_gcp(self, project_id: str, client: monitoring_v3.MetricServiceClient) -> None:
        """Set up GCP project and client."""
        self.project_name = f"projects/{project_id}"
        self.client = client

    def get_metric(self, name: str, metric_type: type[BaseMetric]) -> BaseMetric:
        """Get an existing counter or create and register a new one if it doesn't exist."""
        with self._lock:
            if name in self.registry_map:
                return self.registry_map[name].metric
            metric = metric_type(name)
            self.registry_map[name] = MetricsRegistry(metric)

            return metric

    def export_to_gcp(self) -> None:
        """Export all registered counters to Google Cloud Monitoring."""
        metric_inc("general/metric_manager_export_attempts")

        with self._lock:
            input_list = list(self.registry_map.values())  # Convert the set to a list

        batches: list[list[MetricsRegistry]] = [  #
            input_list[i : i + GCP_EXPORT_BATCH_SIZE] for i in range(0, len(input_list), GCP_EXPORT_BATCH_SIZE)
        ]

        for batch in batches:
            time_series = [s for reg in batch for s in reg.get_series()]
            logging.debug(f"-- MetricManager exporting batch: {[ts.metric.type for ts in time_series]}")
            try:
                assert self.client is not None
                self.client.create_time_series(name=self.project_name, time_series=time_series)
                metric_inc("general/metric_manager_export_success")
            except Exception as e:
                # we will lost all ValueMetrics in the past time interval if this fails
                logging.error(f"-- MetricManager error exporting counters to GCP: {e}")

    async def _periodic_export(self) -> None:
        """Internal method to export metrics periodically."""
        while not self._stop_event.is_set():
            self.export_to_gcp()
            await asyncio.sleep(self.export_interval_secs)

    def start(self) -> None:
        """Start the periodic exporting thread."""
        if self._export_task is None or self._export_task.done():
            self._stop_event.clear()
            self._export_task = asyncio.create_task(self._periodic_export())
            logging.info(f"MetricManager started, GCP Metrics Writing interval: {self.export_interval_secs} s")

    def stop(self) -> None:
        """Stop the periodic exporting thread."""
        self._stop_event.set()
        if self._export_task:
            self._export_task.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(self._export_task)
            except asyncio.CancelledError:
                pass


# initialize this early so it can be used in tests but doesn't depend anything GCP.
METRICS_MANAGER = MetricManager()


def start_metrics_manager() -> None:
    # only initialize everything now.
    global GCM_CLIENT
    GCM_CLIENT = monitoring_v3.MetricServiceClient()

    # only setup anything GCP related now, so we have settings initialized
    METRICS_MANAGER.set_up_gcp(settings.GCP_PROJECT_ID, GCM_CLIENT)
    METRICS_MANAGER.start()


# Convenience functions
def metric_inc(name: str) -> None:
    """increment a named counter by 1"""
    assert METRICS_MANAGER is not None
    m = METRICS_MANAGER.get_metric(name, CounterMetric)
    if isinstance(m, CounterMetric):
        m.inc()
    else:
        raise ValueError(f"Metric {name} is not a CounterMetric")


def metric_inc_by(name: str, amount: int) -> None:
    """increment a named counter by amount"""
    assert METRICS_MANAGER is not None
    m = METRICS_MANAGER.get_metric(name, CounterMetric)
    if isinstance(m, CounterMetric):
        m.inc_by(amount)
    else:
        raise ValueError(f"Metric {name} is not a CounterMetric")


def metric_record(name: str, value: int) -> None:
    """export a specific value for a counter"""
    assert METRICS_MANAGER is not None
    m = METRICS_MANAGER.get_metric(name, ValueMetric)
    if isinstance(m, ValueMetric):
        m.record_value(value)
    else:
        raise ValueError(f"Metric {name} is not a ValueMetric")
