import json
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ypl.backend.llm.reward import (
    RewardClaimedResponse,
    RewardCreationResponse,
    create_reward,
    create_reward_action_log,
    process_reward_claim,
    reward,
)
from ypl.db.rewards import RewardActionLog

router = APIRouter()


@router.post("/rewards/record-action", response_model=RewardCreationResponse)
async def record_reward_action(reward_action_log: RewardActionLog) -> RewardCreationResponse:
    try:
        turn_id = reward_action_log.turn_id

        if turn_id is None:
            raise HTTPException(status_code=400, detail="Turn ID is required")

        updated_reward_action_log = await create_reward_action_log(reward_action_log)

        should_reward, credit_delta, reason = reward(updated_reward_action_log.user_id, turn_id)

        reward_id = None
        if should_reward:
            created_reward = await create_reward(
                user_id=updated_reward_action_log.user_id,
                credit_delta=credit_delta,
                reason=reason,
                reward_action_logs=[updated_reward_action_log],
                turn_id=turn_id,
            )
            reward_id = created_reward.reward_id

        return RewardCreationResponse(is_rewarded=should_reward, reward_id=reward_id)

    except Exception as e:
        log_dict = {
            "message": "Error recording reward action",
            "error": str(e),
        }
        logging.exception(json.dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/reward/{reward_id}/claim", response_model=RewardClaimedResponse)
async def claim_reward(
    reward_id: UUID, user_id: str = Query(..., description="The user ID of the user claiming the reward")
) -> RewardClaimedResponse:
    try:
        reward_claim_struct = await process_reward_claim(reward_id, user_id)

        return RewardClaimedResponse(
            status=reward_claim_struct.status,
            reason=reward_claim_struct.reason,
            credit_delta=reward_claim_struct.credit_delta,
            current_credit_balance=reward_claim_struct.current_credit_balance,
        )

    except Exception as e:
        log_dict = {
            "message": "Error claiming reward",
            "error": str(e),
        }
        logging.exception(json.dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
