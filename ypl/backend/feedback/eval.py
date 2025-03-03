from uuid import UUID

from pydantic import BaseModel
from ypl.backend.db import get_async_session
from ypl.db.chats import Eval, EvalRatingEnum, EvalType, MessageEval


class EvalForMessage(BaseModel):
    """Eval for a single message"""

    message_id: UUID
    rating: EvalRatingEnum
    reasons: list[str] | None = None  # message specific reasons, a list of strings
    comment: str | None = None  # message specific comment


class EvalRequest(BaseModel):
    user_id: UUID
    turn_id: UUID
    eval_type: EvalType
    message_evals: list[EvalForMessage]
    comment: str | None = None  # the overall comment


def _rating_to_score(rating: EvalRatingEnum) -> float:
    """Convert to a score, this is for backwards compatibility"""
    if rating == EvalRatingEnum.GOOD:
        return 100.0
    else:
        return 0.0


async def store_message_eval(eval_req: EvalRequest) -> None:
    async with get_async_session() as session:
        # Add a new eval entry first
        eval = Eval(
            user_id=str(eval_req.user_id),
            turn_id=eval_req.turn_id,
            eval_type=eval_req.eval_type,
            user_comment=eval_req.comment,
        )
        session.add(eval)
        await session.commit()
        await session.refresh(eval)

        for msg_eval in eval_req.message_evals:
            msg_eval_db = MessageEval(
                message_id=msg_eval.message_id,
                eval_id=eval.eval_id,
                score=_rating_to_score(msg_eval.rating),  # for backwards compatibility
                user_comment=",".join(msg_eval.reasons) if msg_eval.reasons else None,
                rating=msg_eval.rating,
                reasons=msg_eval.reasons,
            )
            session.add(msg_eval_db)

        await session.commit()
