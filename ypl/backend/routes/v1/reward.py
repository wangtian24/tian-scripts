from fastapi import APIRouter, HTTPException

from ypl.backend.llm.reward import RewardCreationResponse, create_reward, create_reward_action_log
from ypl.db.rewards import RewardActionLog
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
