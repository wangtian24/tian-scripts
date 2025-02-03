import logging
from dataclasses import dataclass, replace
from datetime import timedelta
from uuid import UUID

import numpy as np
from sqlalchemy import case, func
from sqlmodel import FLOAT, Session, select, update

from ypl.backend.db import get_engine
from ypl.db.chats import ChatMessage
from ypl.db.language_models import LanguageModel


# Metrics object used only locally here.
@dataclass(frozen=True)
class _Metrics:
    model_name: str
    model_id: UUID
    # New metrics calculated with the query
    num_requests: int
    avg_token_count: float
    ttft_p50: float
    ttft_p90: float
    tps_p50: float
    tps_p90: float

    # Existing metrics
    prev_num_requests: int
    prev_avg_token_count: float
    prev_ttft_p50: float
    prev_ttft_p90: float
    prev_tps_p50: float
    prev_tps_p90: float


def update_active_model_metrics(
    metric_window_hours: int = 24,
    max_requests_in_metric_window: int = 100,
    min_requests_in_metric_window: int = 10,
    dry_run: bool = False,
) -> None:
    """
    Updates model metrics in `language_models` table based on up to most recent `max_requests_in_metric_window`
    requests in last `metric_window_hours` hours. This update is run in an hourly cron job.

    Args:
        metric_window_hours : Metrics are based on requests in this many hours in the past.
        max_requests_in_metric_window: Consider up to this main most recent requests
            within the metric window.
        min_requests_in_metric_window: If a model has at least this many requests, its metrics are
            completely based on these requests. Otherwise, the metrics are average of newly
            calculated values and currently existing values. This applies to infrequently used
            models so that very small number of requests don't end up influencing metrics.
        dry_run: Skips updating the table if this is set.
    """

    # Build the query:
    # Calculate model metrics based on recent requests in chat_messages table and join with language_models table.

    recent_chat_messages = (
        # Select chat messages in last metric_window_hours hours. Include row number in descending order of created_at.
        select(
            (ChatMessage.streaming_metrics["requestTimestamp"]).as_float().label("request_ts"),  # type: ignore
            (ChatMessage.streaming_metrics["firstTokenTimestamp"]).as_float().label("first_token_ts"),  # type: ignore
            (ChatMessage.streaming_metrics["lastTokenTimestamp"]).as_float().label("last_token_ts"),  # type: ignore
            (ChatMessage.streaming_metrics["completionTokens"]).as_float().label("token_count"),  # type: ignore
            ChatMessage.assistant_language_model_id.label("model_id"),  # type: ignore
            (
                func.row_number()
                .over(
                    partition_by=ChatMessage.assistant_language_model_id,  # type: ignore
                    order_by=ChatMessage.created_at.desc(),  # type: ignore
                )
                .label("row_num")
            ),
        )
        .filter(
            ChatMessage.message_type == "ASSISTANT_MESSAGE",
            ChatMessage.streaming_metrics.is_not(None),  # type: ignore
            ChatMessage.created_at >= func.now() - timedelta(hours=metric_window_hours),
        )
        .subquery()
    )

    per_message_metrics = (
        # Calculate ttft and tps for each message. Limit messages per model to max_requests_in_metric_window.
        select(
            recent_chat_messages.c.model_id,
            (recent_chat_messages.c.first_token_ts - recent_chat_messages.c.request_ts).label("ttft"),
            case(
                ((recent_chat_messages.c.token_count.is_(None) | (recent_chat_messages.c.token_count == 0)), 0),
                else_=recent_chat_messages.c.token_count
                / (recent_chat_messages.c.last_token_ts - recent_chat_messages.c.request_ts)
                * 1000,
            ).label("tps"),
            recent_chat_messages.c.token_count,
        )
        .where(recent_chat_messages.c.row_num <= max_requests_in_metric_window)
        .subquery()
    )

    model_metrics = (
        # Aggregate metrics per model. Carry all ttft and tps values for percentile calculations.
        select(
            per_message_metrics.c.model_id,
            func.count().label("num_requests"),  # type: ignore
            func.avg(per_message_metrics.c.token_count).cast(FLOAT).label("avg_token_count"),
            func.array_agg(per_message_metrics.c.ttft).label("ttft_values"),
            func.array_agg(per_message_metrics.c.tps).label("tps_values"),
        )
        .filter(per_message_metrics.c.ttft > 0, per_message_metrics.c.tps > 0)
        .group_by(per_message_metrics.c.model_id)
        .subquery()
    )

    model_metrics_joined = (
        # Join with language_models table to get existing metrics.
        select(
            LanguageModel.internal_name.label("model_name"),  # type: ignore
            model_metrics,
            func.coalesce(LanguageModel.num_requests_in_metric_window, 0).label("prev_num_requests"),
            func.coalesce(LanguageModel.first_token_p50_latency_ms, 0.0).label("prev_ttft_p50"),
            func.coalesce(LanguageModel.first_token_p90_latency_ms, 0.0).label("prev_ttft_p90"),
            func.coalesce(LanguageModel.output_p50_tps, 0.0).label("prev_tps_p50"),
            func.coalesce(LanguageModel.output_p90_tps, 0.0).label("prev_tps_p90"),
            func.coalesce(LanguageModel.avg_token_count, 0.0).label("prev_avg_token_count"),
        )
        .join(model_metrics, model_metrics.c.model_id == LanguageModel.language_model_id)
        .order_by(model_metrics.c.num_requests.desc(), LanguageModel.internal_name)
    )

    with Session(get_engine()) as session:
        results = (
            session.exec(model_metrics_joined)
            .mappings()  # returns RowMapping() instead of Row.
            .fetchall()
        )

        logging.info(f"Updating model metrics for {len(results)} models")

        for row in results:
            columns = {**row}
            # Calculate %iles with np
            columns["ttft_p50"], columns["ttft_p90"] = np.percentile(
                np.array(columns["ttft_values"], dtype=float), [50, 90]
            )
            columns["tps_p50"], columns["tps_p90"] = np.percentile(
                np.array(columns["tps_values"], dtype=float), [50, 90]
            )
            del columns["ttft_values"]
            del columns["tps_values"]

            m = _Metrics(**columns)
            if m.num_requests >= min_requests_in_metric_window or m.prev_num_requests == 0:
                # There are enough requests. Overwrite metrics current metrics.
                updated = m
            else:
                # Take average between new and existing metrics.
                # It could be weighted by numer of requests, but probably not any better.
                # TODO: Since these are updated hourly, "previous" metrics here don't really
                #       correspond to metrics from previous day. So this strategy of partially
                #       carrying previous day's metrics does not really work. Models pretty much
                #       end up metrics only based requests 24 hours. To do this better, might need
                #       to save more state or read more requests messages.
                logging.info(
                    f"Not enough requests for {m.model_name} in metric window ({m.num_requests}). "
                    f"Taking average of new and existing values."
                )
                updated = replace(
                    m,
                    ttft_p50=(m.ttft_p50 + m.prev_ttft_p50) / 2,
                    ttft_p90=(m.ttft_p90 + m.prev_ttft_p90) / 2,
                    tps_p50=(m.tps_p50 + m.prev_tps_p50) / 2,
                    tps_p90=(m.tps_p90 + m.prev_tps_p90) / 2,
                    avg_token_count=(m.avg_token_count + m.prev_avg_token_count) / 2,
                )

            # Log and update model table:

            logging.info(
                f"Updating model metrics for {m.model_name} {m.model_id} \t:  "
                f"num requests: {m.num_requests}, "
                f"ttft_p50: {int(m.prev_ttft_p50)} --> {int(updated.ttft_p50)}, "
                f"ttft_p90: {int(m.prev_ttft_p90)} --> {int(updated.ttft_p90)}, "
                f"tps_p50: {m.prev_tps_p50:.2f} --> {updated.tps_p50:.2f}, "
                f"tps_p90: {m.prev_tps_p90:.2f} --> {updated.tps_p90:.2f}, "
                f"avg num tokens {m.prev_avg_token_count} --> {updated.avg_token_count}"
            )

            if not dry_run:
                session.exec(
                    update(LanguageModel)
                    .where(LanguageModel.language_model_id == m.model_id)  # type: ignore
                    .values(
                        first_token_p50_latency_ms=updated.ttft_p50,
                        first_token_p90_latency_ms=updated.ttft_p90,
                        output_p50_tps=updated.tps_p50,
                        output_p90_tps=updated.tps_p90,
                        num_requests_in_metric_window=updated.num_requests,
                        avg_token_count=updated.avg_token_count,
                        # Consider adding 'metrics_updated_at' Timestamp field.
                    )
                )

            # Alternate approach: Store model metrics in a separate table and new rows for each
            # time we update. It can be indexed by creation_time to fetch latest matrics quickly.
            # It is also useful for dashboards.

        if not dry_run:
            session.commit()
            logging.info(f"Updated model metrics for {len(results)} models")
        else:
            logging.info(f"Skipped updating {len(results)} models in this dry run")
