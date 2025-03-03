from datetime import UTC, datetime, timedelta

import humanize
from sqlmodel import select

from ypl.backend.abuse.utils import create_abuse_event
from ypl.backend.db import get_async_session
from ypl.db.abuse import AbuseEventType
from ypl.db.app_feedback import AppFeedback
from ypl.db.chats import Chat, Eval, EvalType, Turn
from ypl.db.users import User

# Time windows for online checks.
SHORT_TIME_WINDOWS = [timedelta(minutes=5), timedelta(minutes=10), timedelta(minutes=30)]

# Time windows for batch/cron checks.
LONG_TIME_WINDOWS = [timedelta(hours=1), timedelta(hours=3), timedelta(hours=6)]

# TODO: this should come from the DB.
# These values reflect the 98th percentile of activities for a sample of users in March 2025.
ACTIVITY_THRESHOLDS = {
    timedelta(minutes=5): {
        "num_chats": 3,
        "num_turns": 7,
        "num_model_feedback": 6,
        "num_qt_feedback": 6,
        "num_downvotes": 5,
        "num_app_feedback": 3,
        "num_any_activity": 10,
    },
    timedelta(minutes=10): {
        "num_chats": 4,
        "num_turns": 8,
        "num_model_feedback": 7,
        "num_qt_feedback": 7,
        "num_downvotes": 6,
        "num_app_feedback": 4,
        "num_any_activity": 18,
    },
    timedelta(minutes=30): {
        "num_chats": 5,
        "num_turns": 16,
        "num_model_feedback": 15,
        "num_qt_feedback": 12,
        "num_downvotes": 8,
        "num_app_feedback": 5,
        "num_any_activity": 50,
    },
    timedelta(hours=1): {
        "num_chats": 7,
        "num_turns": 20,
        "num_model_feedback": 16,
        "num_qt_feedback": 13,
        "num_downvotes": 10,
        "num_app_feedback": 6,
        "num_any_activity": 60,
    },
    timedelta(hours=3): {
        "num_chats": 10,
        "num_turns": 30,
        "num_model_feedback": 20,
        "num_qt_feedback": 15,
        "num_downvotes": 15,
        "num_app_feedback": 10,
        "num_any_activity": 70,
    },
    timedelta(hours=6): {
        "num_chats": 15,
        "num_turns": 40,
        "num_model_feedback": 30,
        "num_qt_feedback": 20,
        "num_downvotes": 20,
        "num_app_feedback": 15,
        "num_any_activity": 80,
    },
}


async def get_recently_active_users(time_window: timedelta) -> set[str]:
    """Returns users that have chat/turn/eval/app_feedback activity within `time_window`."""
    async with get_async_session() as session:
        start_time = datetime.now(UTC) - time_window

        chat_query = select(Chat.creator_user_id).where(Chat.created_at >= start_time)  # type: ignore
        turn_query = select(Turn.creator_user_id).where(Turn.created_at >= start_time)  # type: ignore
        eval_query = select(Eval.user_id).where(Eval.created_at >= start_time)  # type: ignore
        app_feedback_query = select(AppFeedback.user_id).where(AppFeedback.created_at >= start_time)  # type: ignore
        union_query = chat_query.union(turn_query, eval_query, app_feedback_query)

        return set((await session.exec(union_query)).scalars().all())  # type: ignore


async def check_activity_volume_abuse(
    user_id: str,
    end_time: datetime | None = None,
    time_windows: list[timedelta] = LONG_TIME_WINDOWS,
) -> None:
    """Check for activity volume abuse against the thresholds in `ACTIVITY_THRESHOLDS`."""

    for window in time_windows:
        if window not in ACTIVITY_THRESHOLDS:
            raise ValueError(f"Unsupported time window: {window}")
    end_time = end_time or datetime.now(UTC)

    async with get_async_session() as session:
        user = (await session.exec(select(User).where(User.user_id == user_id))).one_or_none()
        if not user:
            raise ValueError(f"User {user_id} not found")

        # Get all events within the longest time window to make just one DB query per activity type.
        longest_time_window = max(time_windows)
        start_time = end_time - longest_time_window

        chat_times_query = select(Chat.created_at).where(
            Chat.creator_user_id == user_id,
            Chat.created_at < end_time,  # type: ignore
            Chat.created_at >= start_time,  # type: ignore
        )
        chat_times = (await session.exec(chat_times_query)).all()

        turn_times_query = select(Turn.created_at).where(
            Turn.creator_user_id == user_id,
            Turn.created_at < end_time,  # type: ignore
            Turn.created_at >= start_time,  # type: ignore
        )
        turn_times = (await session.exec(turn_times_query)).all()

        model_feedback_times = []
        qt_feedback_times = []
        downvote_times = []
        evals_query = select(Eval.created_at, Eval.eval_type).where(  # type: ignore
            Eval.user_id == user_id,
            Eval.created_at < end_time,  # type: ignore
            Eval.created_at >= start_time,  # type: ignore
        )
        for eval_created_at, eval_type in (await session.exec(evals_query)).all():
            match eval_type:
                case EvalType.SELECTION:
                    model_feedback_times.append(eval_created_at)
                case EvalType.QUICK_TAKE:
                    qt_feedback_times.append(eval_created_at)
                case EvalType.DOWNVOTE:
                    downvote_times.append(eval_created_at)

        app_feedback_query = select(AppFeedback.created_at).where(
            AppFeedback.user_id == user_id,
            AppFeedback.created_at < end_time,  # type: ignore
            AppFeedback.created_at >= start_time,  # type: ignore
        )
        app_feedback_times = (await session.exec(app_feedback_query)).all()

        all_times = [
            *chat_times,
            *turn_times,
            *model_feedback_times,
            *qt_feedback_times,
            *downvote_times,
            *app_feedback_times,
        ]

        # Now check each time window separately.
        activities = {}
        for window in time_windows:
            cutoff = end_time - window
            activities[window] = {
                "num_chats": len([t for t in chat_times if t and t >= cutoff]),
                "num_turns": len([t for t in turn_times if t and t >= cutoff]),
                "num_model_feedback": len([t for t in model_feedback_times if t and t >= cutoff]),
                "num_qt_feedback": len([t for t in qt_feedback_times if t and t >= cutoff]),
                "num_downvotes": len([t for t in downvote_times if t and t >= cutoff]),
                "num_app_feedback": len([t for t in app_feedback_times if t and t >= cutoff]),
                "num_any_activity": len([t for t in all_times if t and t >= cutoff]),
            }
            for activity_type in activities[window]:
                threshold = ACTIVITY_THRESHOLDS[window].get(activity_type)
                if threshold and activities[window][activity_type] > threshold:
                    await create_abuse_event(
                        session,
                        user,
                        AbuseEventType.ACTIVITY_VOLUME,
                        event_details={
                            "activity_type": activity_type,
                            "activity_count": activities[window][activity_type],
                            "time_span": humanize.naturaldelta(window),
                            "window_start": start_time,
                            "window_end": end_time,
                            "exceeds_threshold": threshold,
                        },
                    )
