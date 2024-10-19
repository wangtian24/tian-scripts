from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ypl.backend.llm.reward import (
    RewardClaimedResponse,
    RewardCreationResponse,
    RewardUnclaimedResponse,
    create_reward,
    create_reward_action_log,
    process_reward_claim,
)
from ypl.db.rewards import RewardActionLog, RewardStatusEnum
from ypl.logger import logger

router = APIRouter()


@router.post("/rewards/record-action", response_model=RewardCreationResponse)
async def record_reward_action(reward_action_log: RewardActionLog) -> RewardCreationResponse:
    try:
        updated_reward_action_log = await create_reward_action_log(reward_action_log)

        # TODO(ocarmieo): Add logic to check if the user should be rewarded.
        should_reward, credit_delta, reason = True, 10, "THANK_YOU"

        reward_id = None
        if should_reward:
            reward = await create_reward(
                user_id=updated_reward_action_log.user_id,
                credit_delta=credit_delta,
                reason=reason,
                reward_action_logs=[updated_reward_action_log],
            )
            reward_id = reward.reward_id

        return RewardCreationResponse(is_rewarded=should_reward, reward_id=reward_id)

    except Exception as e:
        logger.exception("Error recording reward action: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/reward/{reward_id}/claim", response_model=RewardUnclaimedResponse | RewardClaimedResponse)
async def claim_reward(
    reward_id: UUID, user_id: str = Query(..., description="The user ID of the user claiming the reward")
) -> RewardUnclaimedResponse | RewardClaimedResponse:
    try:
        reward_claim_struct = await process_reward_claim(reward_id, user_id)

        if reward_claim_struct.status == RewardStatusEnum.UNCLAIMED:
            return RewardUnclaimedResponse()
        else:
            return RewardClaimedResponse(
                status=reward_claim_struct.status,
                reason=reward_claim_struct.reason,
                credit_delta=reward_claim_struct.credit_delta,
                current_credit_balance=reward_claim_struct.current_credit_balance,
            )

    except Exception as e:
        logger.exception("Error claiming reward: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
