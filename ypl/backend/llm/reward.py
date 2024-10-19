import logging
import uuid
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import select, update
from sqlmodel.ext.asyncio.session import AsyncSession
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl import logger
from ypl.backend.db import get_async_engine
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.rewards import Reward, RewardActionLog, RewardStatusEnum
from ypl.db.users import User


@dataclass
class RewardCreationResponse:
    is_rewarded: bool = False
    # Reward ID for the client to use while claiming the reward.
    reward_id: UUID | None = None


@dataclass
class RewardUnclaimedResponse:
    status: RewardStatusEnum = RewardStatusEnum.UNCLAIMED


@dataclass
class RewardClaimedResponse:
    reason: str
    credit_delta: int
    current_credit_balance: int
    status: RewardStatusEnum = RewardStatusEnum.CLAIMED


class RewardClaimStruct(BaseModel):
    status: RewardStatusEnum
    reason: str
    credit_delta: int
    current_credit_balance: int


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logger, logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def create_reward_action_log(reward_action_log: RewardActionLog) -> RewardActionLog:
    async with AsyncSession(get_async_engine()) as session:
        session.add(reward_action_log)
        await session.commit()
        await session.refresh(reward_action_log)
        return reward_action_log


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logger, logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def create_reward(
    user_id: str, credit_delta: int, reason: str, reward_action_logs: list[RewardActionLog]
) -> Reward:
    async with AsyncSession(get_async_engine()) as session:
        reward = Reward(user_id=user_id, credit_delta=credit_delta, reason=reason)
        async with session.begin():
            session.add(reward)

            for reward_action_log in reward_action_logs:
                reward_action_log.associated_reward_id = reward.reward_id
                session.add(reward_action_log)

        await session.refresh(reward)
        return reward


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logger, logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def process_reward_claim(reward_id: UUID, user_id: str) -> RewardClaimStruct:
    """
    Processes the reward claim and returns the current credit balance of the user.

    Atomically:
    1. Create a credit transaction for the reward.
    2. Increment the user's credit balance.
    3. Update the reward status to claimed.
    """
    async with AsyncSession(get_async_engine()) as session:
        async with session.begin():
            # Set the isolation level to SERIALIZABLE to prevent concurrent updates of the credit balance.
            await session.connection(execution_options={"isolation_level": "SERIALIZABLE"})
            reward_query = (
                select(Reward, User)
                .join(User)
                .where(
                    Reward.reward_id == reward_id,
                    Reward.deleted_at.is_(None),  # type: ignore
                    Reward.user_id == user_id,
                    Reward.status != RewardStatusEnum.REJECTED,
                )
            )
            result = await session.exec(reward_query)
            reward, user = result.one()

            # Exit early if the reward status is not unclaimed.
            # To keep it idempotent, we return all the info that we get when the reward is newly claimed.
            if reward.status != RewardStatusEnum.UNCLAIMED:
                return RewardClaimStruct(
                    status=reward.status,
                    reason=reward.reason,
                    credit_delta=reward.credit_delta,
                    current_credit_balance=user.points,
                )

            # 1. Create a credit transaction for the reward.
            credit_transaction = PointTransaction(
                transaction_id=uuid.uuid4(),
                user_id=user_id,
                point_delta=reward.credit_delta,
                action_type=PointsActionEnum.REWARD,
                action_details={"reward_id": str(reward.reward_id)},
                claimed_reward_id=reward.reward_id,
            )
            session.add(credit_transaction)

            # 2. Increment the user's credit balance.
            inc_user_credits_stmt = (
                update(User)
                .returning(User.points)  # type: ignore
                .where(
                    User.user_id == user_id,
                    User.deleted_at.is_(None),  # type: ignore
                )
                .values(points=User.points + reward.credit_delta)
            )
            result = await session.exec(inc_user_credits_stmt)
            row = result.one()
            new_credit_balance = int(row.points)  # type: ignore

            # 3. Update the reward status to claimed.
            reward.status = RewardStatusEnum.CLAIMED
            session.add(reward)

        await session.refresh(reward)

        return RewardClaimStruct(
            status=reward.status,
            reason=reward.reason,
            credit_delta=reward.credit_delta,
            current_credit_balance=new_credit_balance,
        )
