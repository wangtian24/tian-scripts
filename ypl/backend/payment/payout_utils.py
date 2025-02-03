import asyncio
import logging
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.orm import selectinload
from sqlmodel import select
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)
from tenacity.asyncio import AsyncRetrying
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.payment.base_types import PaymentInstrumentNotFoundError
from ypl.backend.payment.payment import (
    CashoutPointTransactionRequest,
    PaymentTransactionRequest,
    PointsActionEnum,
    create_cashout_point_transaction,
    create_payment_transaction,
    update_payment_transaction,
    update_user_points,
)
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import fetch_user_names
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrument,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
    PaymentTransaction,
    PaymentTransactionStatusEnum,
)
from ypl.db.point_transactions import PointTransaction

SYSTEM_USER_ID = "SYSTEM"
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 1
RETRY_WAIT_MIN = 4
RETRY_WAIT_MAX = 15
SLACK_WEBHOOK_CASHOUT = settings.SLACK_WEBHOOK_CASHOUT
CASHOUT_TXN_COST = 0


async def async_retry_decorator() -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
        after=after_log(logging.getLogger(), logging.WARNING),
        retry=retry_if_exception_type((OperationalError, DatabaseError)),
    )


async def validate_pending_cashouts_async() -> None:
    """
    Validate pending cashouts.

    This function should be run periodically to check and update the status of pending cashouts.
    It queries for pending payments, verifies them, and updates their status if needed.
    """
    async for attempt in await async_retry_decorator():
        with attempt:
            async with get_async_session() as session:
                query = (
                    select(PaymentTransaction)
                    .options(
                        selectinload(PaymentTransaction.destination_instrument),  # type: ignore
                    )
                    .where(
                        PaymentTransaction.status.not_in(  # type: ignore
                            [
                                PaymentTransactionStatusEnum.SUCCESS,
                                PaymentTransactionStatusEnum.REVERSED,
                                PaymentTransactionStatusEnum.FAILED,
                            ]
                        ),
                        PaymentTransaction.created_at.is_not(None),  # type: ignore
                        PaymentTransaction.created_at < datetime.now() - timedelta(hours=1),  # type: ignore
                        PaymentTransaction.deleted_at.is_(None),  # type: ignore
                    )
                )
                result = await session.execute(query)
                pending_payments = result.scalars().all()
                await session.commit()
                pending_payments_count = len(pending_payments)

                log_dict = {"message": f"Pending payments count: {pending_payments_count}"}
                logging.info(json_dumps(log_dict))

                # for each pending payment capture details and post to slack
                for payment in pending_payments:
                    log_dict = {
                        "message": "Pending payment details",
                        "payment_transaction_id": str(payment.payment_transaction_id),
                        "created_at": str(payment.created_at),
                        "amount": str(payment.amount),
                        "source_instrument_id": str(payment.source_instrument_id),
                        "destination_instrument_id": str(payment.destination_instrument_id),
                        "status": str(payment.status),
                        "partner_reference_id": str(payment.partner_reference_id),
                        "customer_reference_id": str(payment.customer_reference_id),
                    }
                    logging.info(json_dumps(log_dict))

                    facilitator = await get_facilitator_from_source_instrument_id(
                        source_instrument_id=payment.source_instrument_id
                    )
                    if facilitator == PaymentInstrumentFacilitatorEnum.ON_CHAIN:
                        await process_pending_onchain_transaction(payment=payment)
                    elif facilitator == PaymentInstrumentFacilitatorEnum.COINBASE:
                        await process_pending_coinbase_transaction(payment=payment)
                    elif facilitator == PaymentInstrumentFacilitatorEnum.UPI:
                        await process_pending_upi_transaction(payment=payment)
                    else:
                        log_dict = {
                            "message": "Unknown facilitator",
                            "facilitator": str(facilitator),
                        }
                        logging.error(json_dumps(log_dict))


async def get_facilitator_from_source_instrument_id(
    source_instrument_id: uuid.UUID,
) -> PaymentInstrumentFacilitatorEnum | None:
    """
    Get the facilitator from a payment instrument given a source instrument ID.

    Args:
        source_instrument_id: The UUID of the source payment instrument

    Returns:
        PaymentInstrumentFacilitatorEnum: The facilitator of the payment instrument
    """
    async with get_async_session() as session:
        query = select(PaymentInstrument).where(
            PaymentInstrument.payment_instrument_id == source_instrument_id,
            PaymentInstrument.deleted_at.is_(None),  # type: ignore
        )
        result = await session.execute(query)
        payment_instrument = result.scalar_one_or_none()

        if payment_instrument is None:
            log_dict = {
                "message": "Payment instrument not found",
                "source_instrument_id": str(source_instrument_id),
            }
            logging.error(json_dumps(log_dict))
            return None
        facilitator: PaymentInstrumentFacilitatorEnum = payment_instrument.facilitator
        return facilitator


async def process_pending_onchain_transaction(payment: PaymentTransaction) -> None:
    """
    Process an onchain transaction by checking its status and updating the payment transaction record.

    Args:
        payment: The payment transaction to process
    """
    from ypl.backend.payment.crypto.crypto_payout import get_transaction_status

    user_id = payment.destination_instrument.user_id
    try:
        if not payment.partner_reference_id:
            log_dict = {
                "message": ":x: - Missing transaction hash for onchain transaction",
                "payment_transaction_id": str(payment.payment_transaction_id),
            }
            logging.error(json_dumps(log_dict))
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
            return

        tx_status = await get_transaction_status(transaction_hash=payment.partner_reference_id)

        if tx_status == PaymentTransactionStatusEnum.SUCCESS:
            log_dict = {
                "message": ":white_check_mark: - Pending onchain transaction completed",
                "payment_transaction_id": str(payment.payment_transaction_id),
                "transaction_hash": payment.partner_reference_id,
                "old_status": str(payment.status),
                "new_status": str(tx_status),
            }
            logging.info(json_dumps(log_dict))
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
            await update_payment_transaction(
                payment_transaction_id=payment.payment_transaction_id,
                status=PaymentTransactionStatusEnum.SUCCESS,
            )
        elif tx_status == PaymentTransactionStatusEnum.FAILED:
            log_dict = {
                "message": ":x: - Pending onchain transaction failed",
                "payment_transaction_id": str(payment.payment_transaction_id),
                "transaction_hash": payment.partner_reference_id,
            }
            logging.error(json_dumps(log_dict))
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

            points_transaction = await get_points_transaction_from_payment_transaction_id(
                payment.payment_transaction_id
            )
            if points_transaction is None:
                log_dict = {
                    "message": ":x: - No points transaction found for failed payment",
                    "payment_transaction_id": str(payment.payment_transaction_id),
                }
                logging.error(json_dumps(log_dict))
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
                return
            payment_instrument = await get_payment_instrument_from_id(payment.destination_instrument_id)
            if payment_instrument is None:
                log_dict = {
                    "message": ":x: - No payment instrument found for failed payment",
                    "payment_transaction_id": str(payment.payment_transaction_id),
                }
                logging.error(json_dumps(log_dict))
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
                return
            await handle_failed_transaction(
                payment_transaction_id=payment.payment_transaction_id,
                points_transaction_id=points_transaction.transaction_id,
                user_id=str(points_transaction.user_id),
                credits_to_cashout=-points_transaction.point_delta,
                amount=payment.amount,
                source_instrument_id=payment.source_instrument_id,
                destination_instrument_id=payment.destination_instrument_id,
                destination_identifier=payment_instrument.identifier,
                destination_identifier_type=payment_instrument.identifier_type,
                update_points=True,
                currency=payment.currency,
            )
        else:
            log_dict = {
                "message": ":x: - Pending onchain transaction status unknown",
                "payment_transaction_id": str(payment.payment_transaction_id),
                "transaction_hash": payment.partner_reference_id,
            }
            logging.error(json_dumps(log_dict))
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

    except Exception as e:
        log_dict = {
            "message": ":x: - Failed to process pending onchain transaction",
            "payment_transaction_id": str(payment.payment_transaction_id),
            "transaction_hash": payment.partner_reference_id,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))


async def process_pending_coinbase_transaction(payment: PaymentTransaction) -> None:
    """
    Process a Coinbase retail payout transaction by checking its status and updating the payment transaction record.

    Args:
        payment: The payment transaction to process
    """
    from ypl.backend.payment.coinbase.coinbase_facilitator import (
        TransactionStatus,
        get_coinbase_retail_wallet_balance_for_currency,
        get_transaction_status,
    )

    user_id = payment.destination_instrument.user_id
    try:
        if not payment.partner_reference_id:
            log_dict = {
                "message": ":x: - Missing transaction ID for Coinbase retail transaction",
                "payment_transaction_id": str(payment.payment_transaction_id),
            }
            logging.error(json_dumps(log_dict))
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
            return

        # Get the account ID from Coinbase API using the currency
        account_info = await get_coinbase_retail_wallet_balance_for_currency(payment.currency)
        account_id = str(account_info["account_id"])
        if not account_id:
            log_dict = {
                "message": ":x: - Missing account ID for Coinbase retail transaction",
                "payment_transaction_id": str(payment.payment_transaction_id),
                "currency": payment.currency.value,
            }
            logging.error(json_dumps(log_dict))
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
            return

        status = await get_transaction_status(account_id=account_id, transaction_id=payment.partner_reference_id)

        if status == TransactionStatus.COMPLETED.value:
            log_dict = {
                "message": ":white_check_mark: - Pending Coinbase transaction completed",
                "payment_transaction_id": str(payment.payment_transaction_id),
                "transaction_id": payment.partner_reference_id,
                "old_status": str(payment.status),
                "new_status": str(PaymentTransactionStatusEnum.SUCCESS),
            }
            logging.info(json_dumps(log_dict))
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
            await update_payment_transaction(
                payment_transaction_id=payment.payment_transaction_id,
                status=PaymentTransactionStatusEnum.SUCCESS,
            )
        elif status == TransactionStatus.FAILED.value:
            log_dict = {
                "message": ":x: - Pending Coinbase transaction failed",
                "payment_transaction_id": str(payment.payment_transaction_id),
                "transaction_id": payment.partner_reference_id,
            }
            logging.error(json_dumps(log_dict))
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

            points_transaction = await get_points_transaction_from_payment_transaction_id(
                payment.payment_transaction_id
            )
            if points_transaction is None:
                log_dict = {
                    "message": ":x: - No points transaction found for failed payment",
                    "payment_transaction_id": str(payment.payment_transaction_id),
                }
                logging.error(json_dumps(log_dict))
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
                return

            payment_instrument = await get_payment_instrument_from_id(payment.destination_instrument_id)
            if payment_instrument is None:
                log_dict = {
                    "message": ":x: - No payment instrument found for failed payment",
                    "payment_transaction_id": str(payment.payment_transaction_id),
                }
                logging.error(json_dumps(log_dict))
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
                return

            await handle_failed_transaction(
                payment_transaction_id=payment.payment_transaction_id,
                points_transaction_id=points_transaction.transaction_id,
                user_id=str(points_transaction.user_id),
                credits_to_cashout=-points_transaction.point_delta,
                amount=payment.amount,
                source_instrument_id=payment.source_instrument_id,
                destination_instrument_id=payment.destination_instrument_id,
                destination_identifier=payment_instrument.identifier,
                destination_identifier_type=payment_instrument.identifier_type,
                update_points=True,
                currency=payment.currency,
            )
        else:
            log_dict = {
                "message": ":x: - Pending Coinbase transaction status unknown or still pending",
                "payment_transaction_id": str(payment.payment_transaction_id),
                "transaction_id": payment.partner_reference_id,
                "status": status,
            }
            logging.error(json_dumps(log_dict))
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

    except Exception as e:
        log_dict = {
            "message": ":x: - Failed to process pending Coinbase transaction",
            "payment_transaction_id": str(payment.payment_transaction_id),
            "transaction_id": payment.partner_reference_id,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))


async def process_pending_upi_transaction(payment: PaymentTransaction) -> None:
    from ypl.backend.payment.upi.axis.facilitator import AxisUpiFacilitator

    # TODO: Use the base facilitator instead of the specific one
    facilitator: AxisUpiFacilitator = await AxisUpiFacilitator.for_payment_transaction_id(
        payment.payment_transaction_id
    )  # type: ignore
    await facilitator.monitor_payment_status(
        payment_transaction_id=payment.payment_transaction_id,
        partner_reference_id=payment.partner_reference_id,
        user_id=payment.destination_instrument.user_id,
        current_status=payment.status,
        # Retry 6 times, starting with 1s delay, and doubling the delay exponentially up to 10s.
        # Waiting up to ~30s excluding the API call time.
        retry_config={"max_attempts": 6, "multiplier": 1, "min_wait": 1, "max_wait": 10},
    )


async def handle_failed_transaction(
    payment_transaction_id: uuid.UUID,
    points_transaction_id: uuid.UUID | None,
    user_id: str,
    credits_to_cashout: int,
    amount: Decimal,
    source_instrument_id: uuid.UUID,
    destination_instrument_id: uuid.UUID,
    destination_identifier: str,
    destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    update_points: bool,
    currency: CurrencyEnum,
) -> None:
    """Handle cleanup for failed transactions.

    Args:
        payment_transaction_id: The ID of the failed payment transaction
        points_transaction_id: The ID of the points transaction, if one exists
        user_id: The ID of the user
        credits_to_cashout: The number of credits being cashed out
        amount: The amount being transferred
        source_instrument_id: The source payment instrument ID
        destination_instrument_id: The destination payment instrument ID
        destination_identifier: The destination identifier (e.g. wallet address)
        destination_identifier_type: The type of destination identifier
        update_points: Whether to update the user's points
        currency: The currency of the transaction
    """
    try:
        log_dict = {
            "message": ":x: Failure - Failed to process payout reward. Reversing transaction.",
            "user_id": user_id,
            "payment_transaction_id": str(payment_transaction_id),
            "points_transaction_id": str(points_transaction_id),
            "credits_to_cashout": str(credits_to_cashout),
            "amount": str(amount),
            "source_instrument_id": str(source_instrument_id),
            "destination_instrument_id": str(destination_instrument_id),
            "destination_identifier": destination_identifier,
            "destination_identifier_type": destination_identifier_type,
            "currency": currency,
        }
        logging.info(json_dumps(log_dict))
        asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

        await update_payment_transaction(payment_transaction_id, status=PaymentTransactionStatusEnum.FAILED)
        if update_points:
            await update_user_points(user_id, credits_to_cashout)

        reversal_request = PaymentTransactionRequest(
            currency=currency,
            amount=amount,
            source_instrument_id=source_instrument_id,
            destination_instrument_id=destination_instrument_id,
            status=PaymentTransactionStatusEnum.REVERSED,
            additional_info={
                "user_id": user_id,
                "destination_identifier": destination_identifier,
                "destination_identifier_type": str(destination_identifier_type),
                "reversal_transaction_id": str(payment_transaction_id),
            },
        )
        payment_transaction_id = await create_payment_transaction(reversal_request)
        if points_transaction_id:
            points_transaction_id = await create_cashout_point_transaction(
                CashoutPointTransactionRequest(
                    user_id=user_id,
                    point_delta=credits_to_cashout,
                    action_type=PointsActionEnum.CASHOUT_REVERSED,
                    cashout_payment_transaction_id=payment_transaction_id,
                    action_details={
                        "reversal_payment_transaction_id": str(payment_transaction_id),
                    },
                )
            )
        log_dict = {
            "message": ":white_check_mark: Success - Reversed transaction",
            "payment_transaction_id": str(payment_transaction_id),
            "points_transaction_id": str(points_transaction_id),
            "user_id": user_id,
            "amount": str(amount),
            "source_instrument_id": str(source_instrument_id),
            "destination_instrument_id": str(destination_instrument_id),
            "destination_identifier": destination_identifier,
            "destination_identifier_type": destination_identifier_type,
        }
        logging.info(json_dumps(log_dict))
        asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
    except Exception as e:
        error_message = str(e)
        log_dict = {
            "message": ":x: Failure - Failed to handle failed transaction cleanup",
            "payment_transaction_id": str(payment_transaction_id),
            "points_transaction_id": str(points_transaction_id),
            "user_id": user_id,
            "amount": str(amount),
            "source_instrument_id": str(source_instrument_id),
            "destination_instrument_id": str(destination_instrument_id),
            "destination_identifier": destination_identifier,
            "destination_identifier_type": destination_identifier_type,
            "error": error_message,
        }
        logging.exception(json_dumps(log_dict))
        asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))


async def get_points_transaction_from_payment_transaction_id(
    payment_transaction_id: uuid.UUID,
) -> PointTransaction | None:
    """Get the points transaction associated with a payment transaction.

    Args:
        payment_transaction_id: The UUID of the payment transaction

    Returns:
        PointTransaction | None: The points transaction if found, None otherwise
    """
    async with get_async_session() as session:
        query = select(PointTransaction).where(
            PointTransaction.cashout_payment_transaction_id == payment_transaction_id,
        )
        result = await session.execute(query)
        points_transaction: PointTransaction | None = result.scalar_one_or_none()
        return points_transaction


async def get_payment_instrument_from_id(payment_instrument_id: uuid.UUID) -> PaymentInstrument | None:
    """Get a payment instrument by its ID.

    Args:
        payment_instrument_id: The UUID of the payment instrument

    Returns:
        PaymentInstrument | None: The payment instrument if found, None otherwise
    """
    async with get_async_session() as session:
        query = select(PaymentInstrument).where(
            PaymentInstrument.payment_instrument_id == payment_instrument_id,
            PaymentInstrument.deleted_at.is_(None),  # type: ignore
        )
        result = await session.execute(query)
        payment_instrument: PaymentInstrument | None = result.scalar_one_or_none()
        return payment_instrument


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
)
async def get_source_instrument_id(
    facilitator: PaymentInstrumentFacilitatorEnum,
    identifier_type: PaymentInstrumentIdentifierTypeEnum,
    identifier: str | None = None,
) -> uuid.UUID:
    """Get the source payment instrument ID for a given facilitator and identifier type.

    Args:
        facilitator: The payment facilitator type
        identifier_type: The type of identifier for the payment instrument
        identifier: The identifier for the payment instrument to filter on, if provided

    Returns:
        UUID: The payment instrument ID

    Raises:
        PaymentInstrumentNotFoundError: If no payment instrument is found
    """
    async with get_async_session() as session:
        query = select(PaymentInstrument).where(
            PaymentInstrument.facilitator == facilitator,
            PaymentInstrument.identifier_type == identifier_type,
            PaymentInstrument.user_id == SYSTEM_USER_ID,
            PaymentInstrument.identifier == identifier if identifier is not None else True,
            PaymentInstrument.deleted_at.is_(None),  # type: ignore
        )

        result = await session.exec(query)
        instrument = result.first()
        if not instrument:
            log_dict = {
                "message": "Source payment instrument not found",
                "identifier_type": identifier_type,
                "facilitator": facilitator,
                "user_id": SYSTEM_USER_ID,
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentInstrumentNotFoundError(f"Payment instrument not found for {facilitator}")
        return instrument.payment_instrument_id


async def get_destination_instrument_id(
    facilitator: PaymentInstrumentFacilitatorEnum,
    user_id: str,
    destination_identifier: str,
    destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    instrument_metadata: dict | None = None,
) -> uuid.UUID:
    """Get or create the destination payment instrument ID.

    Args:
        facilitator: The payment facilitator type
        user_id: The ID of the user
        destination_identifier: The destination identifier (e.g. wallet address)
        destination_identifier_type: The type of identifier for the payment instrument
        instrument_metadata: The metadata for the payment instrument

    Returns:
        UUID: The payment instrument ID
    """
    async with get_async_session() as session:
        query = select(PaymentInstrument).where(
            PaymentInstrument.facilitator == facilitator,
            PaymentInstrument.identifier_type == destination_identifier_type,
            func.lower(PaymentInstrument.identifier) == destination_identifier.lower(),
            PaymentInstrument.deleted_at.is_(None),  # type: ignore
        )
        result = await session.exec(query)
        existing_instruments = result.all()

        instrument: PaymentInstrument

        if not existing_instruments:
            log_dict = {
                "message": "Destination payment instrument not found. Creating a new one.",
                "identifier_type": destination_identifier_type,
                "facilitator": facilitator,
                "user_id": user_id,
                "identifier": destination_identifier,
                "instrument_metadata": instrument_metadata,
            }
            logging.info(json_dumps(log_dict))
            instrument = PaymentInstrument(
                facilitator=facilitator,
                identifier_type=destination_identifier_type,
                identifier=destination_identifier,
                user_id=user_id,
                instrument_metadata=instrument_metadata,
            )
            session.add(instrument)
        else:
            user_instrument = next((i for i in existing_instruments if i.user_id == user_id), None)

            if not user_instrument:
                existing_user_ids = [i.user_id for i in existing_instruments]
                existing_user_names = await fetch_user_names(existing_user_ids)
                log_dict = {
                    "message": ":warning: - Payment instrument reuse attempt",
                    "new_user_id": user_id,
                    "existing_user_ids": existing_user_ids,
                    "existing_user_names": list(existing_user_names.values()),
                    "identifier": destination_identifier,
                    "identifier_type": destination_identifier_type,
                    "facilitator": facilitator,
                    "count_of_users_already_using_instrument": len(existing_instruments),
                }
                logging.warning(json_dumps(log_dict))
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

                # Allow reuse if only one user is currently using the instrument
                if len(existing_instruments) < 2:
                    log_dict = {
                        "message": "Creating new payment instrument for user (instrument exists for one other user)",
                        "identifier_type": destination_identifier_type,
                        "facilitator": facilitator,
                        "user_id": user_id,
                        "existing_user_id": existing_user_ids[0],
                        "existing_user_name": existing_user_names[existing_user_ids[0]],
                        "identifier": destination_identifier,
                        "instrument_metadata": instrument_metadata,
                    }
                    logging.info(json_dumps(log_dict))
                    instrument = PaymentInstrument(
                        facilitator=facilitator,
                        identifier_type=destination_identifier_type,
                        identifier=destination_identifier,
                        user_id=user_id,
                        instrument_metadata=instrument_metadata,
                    )
                    session.add(instrument)
                else:
                    # If 2 or more users are already using this instrument, raise exception
                    log_dict = {
                        "message": ":x: Payment instrument reuse attempt for more than allowed users",
                        "new_user_id": user_id,
                        "existing_user_ids": existing_user_ids,
                        "existing_user_names": list(existing_user_names.values()),
                        "identifier": destination_identifier,
                        "identifier_type": destination_identifier_type,
                        "facilitator": facilitator,
                    }
                    logging.info(json_dumps(log_dict))
                    asyncio.create_task(
                        post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT)
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "This payment info is already linked to two other users. "
                            "We don't support sharing payment destinations across more than two users"
                        ),
                    )
            else:
                instrument = user_instrument
                if instrument_metadata is not None:
                    log_dict = {
                        "message": "Updating instrument metadata",
                        "instrument_id": str(instrument.payment_instrument_id),
                        "old_metadata": instrument.instrument_metadata,
                        "new_metadata": instrument_metadata,
                        "user_id": user_id,
                        "identifier": destination_identifier,
                        "identifier_type": destination_identifier_type,
                        "facilitator": facilitator,
                    }
                    logging.info(json_dumps(log_dict))
                    if instrument.instrument_metadata is None:
                        instrument.instrument_metadata = instrument_metadata
                    else:
                        new_metadata = deepcopy(instrument.instrument_metadata or {})
                        new_metadata.update(instrument_metadata)
                        instrument.instrument_metadata = new_metadata

        await session.commit()
        return instrument.payment_instrument_id


async def get_instrument_identifier(instrument_id: uuid.UUID) -> str | None:
    """Get the instrument identifier for a given instrument ID."""
    async with get_async_session() as session:
        query = select(PaymentInstrument).where(PaymentInstrument.payment_instrument_id == instrument_id)
        result = await session.execute(query)
        instrument = result.scalar_one_or_none()
        return instrument.identifier if instrument else None
