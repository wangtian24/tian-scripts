import logging
from datetime import datetime, timedelta

import humanize
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.abuse.utils import create_abuse_event, get_user_events
from ypl.backend.db import get_async_session
from ypl.backend.llm.turn_quality import LOW_EVAL_QUALITY_SCORE
from ypl.db.abuse import AbuseEventType
from ypl.db.chats import Eval, EvalType
from ypl.db.users import User

# The minimum number of quality scores to consider for abuse detection.
MIN_NUM_QUALITY_SCORES = 5
# If the user has more than this number of consecutive low quality scores, they are abusing the model feedback feature.
CONSECUTIVE_LOW_QUALITY_THRESHOLD = 3
# If the user has more than this fraction of low quality scores, they are abusing the model feedback feature.
FRACTION_LOW_QUALITY_THRESHOLD = 0.4


async def get_recent_eval_quality_scores(
    user_id: str, session: AsyncSession, num_recent_evals: int, end_time: datetime | None = None
) -> list[float]:
    query = (
        select(Eval.quality_score)
        .where(
            Eval.user_id == user_id,
            Eval.deleted_at.is_(None),  # type: ignore
            Eval.eval_type == EvalType.SELECTION,
        )
        .order_by(Eval.created_at.desc())  # type: ignore
        .limit(num_recent_evals)
    )
    if end_time:
        query = query.where(Eval.created_at <= end_time)  # type: ignore

    result = await session.exec(query)
    scores = result.all()
    return [score for score in scores if score is not None]


def _low_quality_score_sequence_reason(scores: list[float]) -> str | None:
    """Checks the sequence for the overall fraction of low quality scores and consecutive runs of them."""

    if len(scores) < MIN_NUM_QUALITY_SCORES:
        return None

    total_low_quality = 0
    max_consecutive_low_quality = 0
    consecutive_low_quality = 0
    for score in scores:
        if score == LOW_EVAL_QUALITY_SCORE:
            total_low_quality += 1
            consecutive_low_quality += 1
            max_consecutive_low_quality = max(max_consecutive_low_quality, consecutive_low_quality)
        else:
            # reset
            consecutive_low_quality = 0

    if total_low_quality > FRACTION_LOW_QUALITY_THRESHOLD * len(scores):
        return f"too many low quality evals ({total_low_quality}) in the last {len(scores)} ones"
    if max_consecutive_low_quality > CONSECUTIVE_LOW_QUALITY_THRESHOLD:
        return f"too many consecutive low quality evals ({max_consecutive_low_quality}) in the last {len(scores)}"

    return None


async def get_model_feedback_time_deltas(
    user_id: str, end_time: datetime | None = None, limit: int = 10
) -> list[timedelta]:
    """Returns recent time differences between end of model streaming and model preferences."""
    events = await get_user_events(user_id, end_time=end_time, event_names=["prefer_this", "streaming_complete"])
    deltas: list[timedelta] = []
    prev_event_name, prev_event_time = "", None
    for event in events:
        event_name, event_time = event[0], event[3]
        if event_time and prev_event_time and event_name == "streaming_complete" and prev_event_name == "prefer_this":
            deltas.append(event_time - prev_event_time)
        prev_event_name, prev_event_time = event_name, event_time

    return deltas[:limit]


async def check_model_feedback_abuse(user_id: str, end_time: datetime | None = None) -> None:
    try:
        async with get_async_session() as session:
            user = (await session.exec(select(User).where(User.user_id == user_id))).one_or_none()
            if not user:
                raise ValueError(f"User {user_id} not found")

            scores = await get_recent_eval_quality_scores(user_id, session, num_recent_evals=10, end_time=end_time)
            reason = _low_quality_score_sequence_reason(scores)
            if reason:
                model_feedback_time_deltas = await get_model_feedback_time_deltas(user_id, end_time=end_time)
                await create_abuse_event(
                    session,
                    user,
                    AbuseEventType.CONTENT_LOW_QUALITY_MODEL_FEEDBACK,
                    event_details={
                        "reason": reason,
                        "recent_model_feedback_quality_scores": scores,
                        "recent_model_feedback_response_times": [
                            humanize.naturaldelta(delta) for delta in model_feedback_time_deltas
                        ],
                    },
                )
    except Exception as e:
        logging.error(f"Error checking model feedback abuse for user {user_id}: {e}")
