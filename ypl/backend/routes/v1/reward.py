import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ypl.backend.llm.chat import get_turn_id_from_message_id
from ypl.backend.llm.reward import (
    RewardAmountRule,
    RewardClaimedResponse,
    RewardCreationResponse,
    RewardProbabilityRule,
    create_reward,
    create_reward_action_log,
    feedback_based_reward,
    get_reward_action_log_by_user_and_turn,
    process_reward_claim,
    qt_eval_reward,
    turn_based_reward,
)
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.db.rewards import RewardActionEnum, RewardActionLog

router = APIRouter()


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


async def process_reward_creation_and_claim(
    user_id: str,
    credit_delta: int,
    comment: str,
    reward_action_log: RewardActionLog,
    turn_id: UUID | None,
    reward_amount_rule: RewardAmountRule | None,
    reward_probability_rule: RewardProbabilityRule | None,
) -> RewardCreationResponse:
    """Process reward creation and claim synchronously.

    Args:
        user_id: The ID of the user receiving the reward
        credit_delta: Amount of credits to award
        comment: Description of the reward
        reward_action_log: Log of the action that triggered the reward
        turn_id: Optional ID of the conversation turn (not used for feedback)
        reward_amount_rule: Rule used to calculate reward amount
        reward_probability_rule: Rule used to determine reward probability

    Returns:
        RewardCreationResponse with reward status and details

    Raises:
        Exception: If reward creation or claiming fails
    """
    try:
        # Create reward synchronously
        created_reward = await create_reward(
            user_id=user_id,
            credit_delta=credit_delta,
            comment=comment,
            reward_action_logs=[reward_action_log],
            turn_id=turn_id,
            reward_amount_rule=reward_amount_rule,
            reward_probability_rule=reward_probability_rule,
        )

        # Process claim synchronously
        reward_claim = await process_reward_claim(created_reward.reward_id, user_id)

        return RewardCreationResponse(
            is_rewarded=True,
            reward_id=created_reward.reward_id,
            comment=reward_claim.comment,
            credit_delta=credit_delta,
        )

    except Exception as e:
        log_dict = {
            "message": "Error in process_reward_creation_and_claim",
            "user_id": user_id,
            "credit_delta": credit_delta,
            "turn_id": str(turn_id) if turn_id else None,
            "reward_action_log": reward_action_log.dict(),
            "error": str(e),
        }
        logging.exception("Reward processing failed", extra=log_dict)

        # Background notification using context manager
        asyncio.create_task(notify_slack_error(user_id, credit_delta, str(e)))

        return RewardCreationResponse(is_rewarded=False)


async def handle_feedback_reward(reward_action_log: RewardActionLog) -> RewardCreationResponse:
    """Handle feedback-based reward processing."""
    if not reward_action_log.action_details or "feedback_comment" not in reward_action_log.action_details:
        return RewardCreationResponse(is_rewarded=False, credit_delta=0)

    updated_reward_action_log = await create_reward_action_log(reward_action_log)

    should_reward, credit_delta, comment, amount_rule, prob_rule = await feedback_based_reward(
        updated_reward_action_log.user_id,
        reward_action_log.action_details["feedback_comment"],
    )

    if not should_reward:
        return RewardCreationResponse(is_rewarded=False, credit_delta=0)

    return await process_reward_creation_and_claim(
        user_id=updated_reward_action_log.user_id,
        credit_delta=credit_delta,
        comment=comment,
        reward_action_log=updated_reward_action_log,
        turn_id=None,
        reward_amount_rule=amount_rule,
        reward_probability_rule=prob_rule,
    )


async def handle_qt_eval_reward(reward_action_log: RewardActionLog) -> RewardCreationResponse:
    """Handle QT (Quick Take) evaluation reward processing."""
    # Check for message_id in action_details
    if not reward_action_log.action_details or "message_id" not in reward_action_log.action_details:
        raise HTTPException(status_code=400, detail="Message ID is required in action_details for QT eval actions")

    message_id = UUID(reward_action_log.action_details["message_id"])

    turn_id = await get_turn_id_from_message_id(message_id)
    if turn_id is None:
        raise HTTPException(status_code=404, detail="Could not find turn_id for the given message_id")

    # Check if an entry already exists for this user and turn
    exists_reward_action_log = await get_reward_action_log_by_user_and_turn(
        user_id=reward_action_log.user_id,
        turn_id=turn_id,
        action_type=RewardActionEnum.QT_EVAL.name,
    )

    # do not reward the user for multiple QT eval actions in a single turn
    if exists_reward_action_log:
        return RewardCreationResponse(is_rewarded=False, credit_delta=0)

    updated_reward_action_log = await create_reward_action_log(reward_action_log)
    should_reward, credit_delta, comment, reward_amount_rule, reward_probability_rule = await qt_eval_reward(
        updated_reward_action_log.user_id
    )

    if should_reward:
        return await process_reward_creation_and_claim(
            user_id=updated_reward_action_log.user_id,
            credit_delta=credit_delta,
            comment=comment,
            reward_action_log=updated_reward_action_log,
            turn_id=turn_id,
            reward_amount_rule=reward_amount_rule,
            reward_probability_rule=reward_probability_rule,
        )

    return RewardCreationResponse(is_rewarded=False, credit_delta=0)


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
            "user_id": reward_action_log.user_id,
            "reward_action_log": reward_action_log,
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


async def notify_slack_error(user_id: str, credit_delta: int, error: str) -> None:
    """Send error notification to Slack asynchronously."""
    message = (
        f":x: Reward creation and claim failed\n" f"User: {user_id}\n" f"Amount: {credit_delta}\n" f"Error: {error}"
    )
    await post_to_slack(message)
