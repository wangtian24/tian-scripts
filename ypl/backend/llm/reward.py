import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel.ext.asyncio.session import AsyncSession
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl import logger
from ypl.backend.db import get_async_engine
from ypl.db.rewards import Reward, RewardActionLog


@dataclass
class RewardCreationResponse:
    is_rewarded: bool = False
    # Reward ID for the client to use while claiming the reward.
    reward_id: UUID | None = None


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
