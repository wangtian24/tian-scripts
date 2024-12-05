import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ypl.backend.llm.reward import (
    RewardClaimedResponse,
    RewardCreationResponse,
    create_reward,
    create_reward_action_log,
    feedback_based_reward,
    get_reward_action_log_by_user_and_turn,
    process_reward_claim,
    qt_eval_reward,
    turn_based_reward,
)
from ypl.backend.utils.json import json_dumps
from ypl.db.rewards import RewardActionEnum, RewardActionLog

router = APIRouter()


async def handle_feedback_reward(reward_action_log: RewardActionLog) -> RewardCreationResponse:
    """Handle feedback-based reward processing."""
    updated_reward_action_log = await create_reward_action_log(reward_action_log)
    should_reward, credit_delta, comment, reward_amount_rule, reward_probability_rule = feedback_based_reward(
        updated_reward_action_log.user_id
    )

    if should_reward:
        created_reward = await create_reward(
            user_id=updated_reward_action_log.user_id,
            credit_delta=credit_delta,
            comment=comment,
            reward_action_logs=[updated_reward_action_log],
            turn_id=None,
            reward_amount_rule=reward_amount_rule,
            reward_probability_rule=reward_probability_rule,
        )

        # process reward claim always for feedback-based rewards as users will not do a scratchcard for kabini release
        # TODO post kabini release, we should send scratchcards and not automatically claim rewards
        reward_claim_struct = await process_reward_claim(created_reward.reward_id, updated_reward_action_log.user_id)

        return RewardCreationResponse(
            is_rewarded=True,
            reward_id=created_reward.reward_id,
            comment=reward_claim_struct.comment,
            credit_delta=credit_delta,
        )

    return RewardCreationResponse(is_rewarded=False)


async def handle_turn_reward(reward_action_log: RewardActionLog) -> RewardCreationResponse:
    """Handle turn-based reward processing."""
    turn_id = reward_action_log.turn_id
    if turn_id is None:
        raise HTTPException(status_code=400, detail="Turn ID is required for non-feedback actions")

    updated_reward_action_log = await create_reward_action_log(reward_action_log)
    should_reward, credit_delta, comment, reward_amount_rule, reward_probability_rule = turn_based_reward(
        updated_reward_action_log.user_id, turn_id
    )

    if should_reward:
        created_reward = await create_reward(
            user_id=updated_reward_action_log.user_id,
            credit_delta=credit_delta,
            comment=comment,
            reward_action_logs=[updated_reward_action_log],
            turn_id=turn_id,
            reward_amount_rule=reward_amount_rule,
            reward_probability_rule=reward_probability_rule,
        )

        return RewardCreationResponse(
            is_rewarded=True, reward_id=created_reward.reward_id, comment=comment, credit_delta=credit_delta
        )

    return RewardCreationResponse(is_rewarded=False)


async def handle_qt_eval_reward(reward_action_log: RewardActionLog) -> RewardCreationResponse:
    """Handle QT (Quick Take) evaluation reward processing."""
    turn_id = reward_action_log.turn_id
    if turn_id is None:
        raise HTTPException(status_code=400, detail="Turn ID is required for QT eval actions")

    # Check if an entry already exists for this user and turn
    existing_log = await get_reward_action_log_by_user_and_turn(
        user_id=reward_action_log.user_id,
        turn_id=turn_id,
    )

    # do not reward the user for multiple QT eval actions in a single turn
    if existing_log:
        return RewardCreationResponse(is_rewarded=False)

    updated_reward_action_log = await create_reward_action_log(reward_action_log)
    should_reward, credit_delta, comment, reward_amount_rule, reward_probability_rule = qt_eval_reward(
        updated_reward_action_log.user_id
    )

    if should_reward:
        created_reward = await create_reward(
            user_id=updated_reward_action_log.user_id,
            credit_delta=credit_delta,
            comment=comment,
            reward_action_logs=[updated_reward_action_log],
            turn_id=turn_id,
            reward_amount_rule=reward_amount_rule,
            reward_probability_rule=reward_probability_rule,
        )

        # process reward claim always for QT Eval rewards as users will not do a scratchcard for kabini release
        # TODO post kabini release, we should send scratchcards and not automatically claim rewards
        reward_claim_struct = await process_reward_claim(created_reward.reward_id, updated_reward_action_log.user_id)

        return RewardCreationResponse(
            is_rewarded=True,
            reward_id=created_reward.reward_id,
            comment=reward_claim_struct.comment,
            credit_delta=credit_delta,
        )

    return RewardCreationResponse(is_rewarded=False)


@router.post("/rewards/record-action", response_model=RewardCreationResponse)
async def record_reward_action(reward_action_log: RewardActionLog) -> RewardCreationResponse:
    try:
        if reward_action_log.action_type == RewardActionEnum.FEEDBACK.name:
            return await handle_feedback_reward(reward_action_log)
        elif reward_action_log.action_type == RewardActionEnum.QT_EVAL.name:
            return await handle_qt_eval_reward(reward_action_log)
        else:
            return await handle_turn_reward(reward_action_log)

    except Exception as e:
        log_dict = {
            "message": "Error recording reward action",
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/reward/{reward_id}/claim", response_model=RewardClaimedResponse)
async def claim_reward(
    reward_id: UUID, user_id: str = Query(..., description="The user ID of the user claiming the reward")
) -> RewardClaimedResponse:
    try:
        reward_claim_struct = await process_reward_claim(reward_id, user_id)

        return RewardClaimedResponse(
            status=reward_claim_struct.status,
            comment=reward_claim_struct.comment,
            # TODO(arawind): Stop populating reason.
            reason=reward_claim_struct.comment,
            credit_delta=reward_claim_struct.credit_delta,
            current_credit_balance=reward_claim_struct.current_credit_balance,
        )

    except Exception as e:
        log_dict = {
            "message": "Error claiming reward",
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
