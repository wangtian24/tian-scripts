from sqlalchemy import select
from sqlmodel import Session

from ypl.backend.db import get_engine
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.users import User


def reset_points(init_value: int = 100):
    with Session(get_engine()) as session:
        yuppsters = session.exec(select(User).where(User.email.like("%@yupp.ai"))).scalars().all()

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
            print(f"{user.email:>20}:\t{reason}")

        session.commit()
