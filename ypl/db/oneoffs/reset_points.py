import logging

from sqlalchemy import select
from sqlmodel import Session

from ypl.backend.db import get_engine
from ypl.backend.utils.json import json_dumps
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.users import User


def reset_points(init_value: int = 1500) -> None:
    with Session(get_engine()) as session:
        stmt = select(User).where(User.email.like("%@yupp.ai"))  # type: ignore[attr-defined]
        yuppsters = session.exec(stmt).scalars().all()  # type: ignore[call-overload]

        for user in yuppsters:
            if user.points <= init_value:
                continue

            reason = f"Reset points from {user.points} to {init_value}"
            point_delta = init_value - user.points
            user.points = init_value

            adjustment = PointTransaction(
                user_id=user.user_id,
                point_delta=point_delta,
                action_type=PointsActionEnum.ADJUSTMENT,
                action_details={"adjustment_reason": reason},
            )
            session.add(adjustment)
            log_dict = {
                "message": "Weekly reset of points for Yuppster",
                "user_id": user.user_id,
                "user_email": user.email,
                "point_delta": point_delta,
                "action_type": PointsActionEnum.ADJUSTMENT,
                "action_details": {"adjustment_reason": reason},
            }
            logging.info(json_dumps(log_dict))

        session.commit()
