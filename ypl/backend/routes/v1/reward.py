import asyncio
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
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.db.rewards import RewardActionEnum, RewardActionLog

router = APIRouter()

MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


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


async def handle_feedback_reward(reward_action_log: RewardActionLog) -> RewardCreationResponse:
    """Handle feedback-based reward processing."""

    # check if an out of range reward value has been passed in action details
    # This is to ensure that incase frontend decided the reward, it is within the bounds
    if reward_action_log.action_details and "reward_amount" in reward_action_log.action_details:
        reward_amount = float(reward_action_log.action_details["reward_amount"])
    else:
        reward_amount = None

    updated_reward_action_log = await create_reward_action_log(reward_action_log)
    should_reward, credit_delta, comment, reward_amount_rule, reward_probability_rule = feedback_based_reward(
        updated_reward_action_log.user_id, reward_amount
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

        # TODO post kabini release, we should send scratchcards and not automatically claim rewards
        # Create task for process_reward_claim and add error callback
        task_name = (
            f"function: handle_feedback_reward "
            f"reward_claim for user: {updated_reward_action_log.user_id} "
            f"with reward_id: {created_reward.reward_id}"
        )
        task = asyncio.create_task(
            retry_reward_claim(created_reward.reward_id, updated_reward_action_log.user_id), name=task_name
        )
        task.add_done_callback(handle_background_task_error)

        return RewardCreationResponse(
            is_rewarded=True,
            reward_id=created_reward.reward_id,
            comment=comment,
            credit_delta=credit_delta,
        )

    return RewardCreationResponse(is_rewarded=False)


async def handle_qt_eval_reward(reward_action_log: RewardActionLog) -> RewardCreationResponse:
    """Handle QT (Quick Take) evaluation reward processing."""
    turn_id = reward_action_log.turn_id
    if turn_id is None:
        raise HTTPException(status_code=400, detail="Turn ID is required for QT eval actions")

    # check if an out of range reward value has been passed in action details
    if reward_action_log.action_details and "reward_amount" in reward_action_log.action_details:
        reward_amount = float(reward_action_log.action_details["reward_amount"])
    else:
        reward_amount = None

    # Check if an entry already exists for this user and turn
    existing_log = await get_reward_action_log_by_user_and_turn(
        user_id=reward_action_log.user_id,
        turn_id=turn_id,
        action_type=RewardActionEnum.QT_EVAL.name,
    )

    # do not reward the user for multiple QT eval actions in a single turn
    if existing_log:
        return RewardCreationResponse(is_rewarded=False)

    updated_reward_action_log = await create_reward_action_log(reward_action_log)
    should_reward, credit_delta, comment, reward_amount_rule, reward_probability_rule = qt_eval_reward(
        updated_reward_action_log.user_id, reward_amount
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

        # TODO post kabini release, we should send scratchcards and not automatically claim rewards
        # Create task for process_reward_claim and add error callback
        task_name = (
            f"handle_qt_eval_reward reward_claim "
            f"for user: {updated_reward_action_log.user_id} "
            f"with reward_id: {created_reward.reward_id}"
        )
        task = asyncio.create_task(
            retry_reward_claim(created_reward.reward_id, updated_reward_action_log.user_id), name=task_name
        )
        task.add_done_callback(handle_background_task_error)

        return RewardCreationResponse(
            is_rewarded=True,
            reward_id=created_reward.reward_id,
            comment=comment,
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


async def retry_reward_claim(reward_id: UUID, user_id: str, attempt: int = 1) -> None:
    """
    Retry the reward claim with exponential backoff.
    """
    try:
        await process_reward_claim(reward_id, user_id)
    except Exception as exc:
        if attempt >= MAX_RETRIES:
            logging.error(
                json_dumps(
                    {
                        "message": f"Reward claim failed after {MAX_RETRIES} attempts",
                        "error": str(exc),
                        "user_id": user_id,
                        "reward_id": reward_id,
                    }
                )
            )
            return

        # Exponential backoff
        wait_time = RETRY_DELAY * (2 ** (attempt - 1))
        logging.warning(
            json_dumps(
                {
                    "message": f"Reward claim attempt {attempt} failed, retrying in {wait_time}s",
                    "error": str(exc),
                    "user_id": user_id,
                    "reward_id": reward_id,
                }
            )
        )
        await asyncio.sleep(wait_time)
        await retry_reward_claim(reward_id, user_id, attempt + 1)


def handle_background_task_error(task: asyncio.Task) -> None:
    """Handle any errors from background tasks."""
    try:
        # Get the exception if the task failed
        exc = task.exception()
        if exc:
            log_dict = {
                "credit_claim_failed": True,
                "message": task.get_name(),
                "error": str(exc),
                "user_id": task.get_name().split("user: ")[1].split(" ")[0],  # Extract from task name
                "reward_id": task.get_name().split("reward_id: ")[1],  # Extract from task name
            }
            logging.exception(json_dumps(log_dict))

            # Create a new async task for Slack notification
            slack_message = (
                f":x: Reward claim failed for user {log_dict['user_id']}\n"
                f"Reward ID: {log_dict['reward_id']}\n"
                f"Error: {str(exc)}"
            )
            # Create a new task for the async Slack call
            asyncio.create_task(post_to_slack(slack_message))

    except asyncio.CancelledError:
        pass
