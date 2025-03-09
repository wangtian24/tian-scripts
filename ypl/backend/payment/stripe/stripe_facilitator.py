import asyncio
import logging
import os
import time
import uuid
from decimal import Decimal
from uuid import UUID

from sqlmodel import select
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.payment.facilitator import (
    BaseFacilitator,
    PaymentInstrumentError,
    PaymentProcessingError,
    PaymentResponse,
    PaymentStatusFetchError,
)
from ypl.backend.payment.payment import (
    CashoutPointTransactionRequest,
    PaymentTransactionRequest,
    create_cashout_point_transaction,
    create_payment_transaction,
    update_payment_transaction,
    update_user_points,
)
from ypl.backend.payment.payout_utils import (
    CASHOUT_TXN_COST,
    handle_failed_transaction,
)
from ypl.backend.payment.payout_utils import (
    get_destination_instrument as get_generic_destination_instrument,
)
from ypl.backend.payment.payout_utils import (
    get_source_instrument as get_generic_source_instrument,
)
from ypl.backend.payment.stripe.stripe_payout import (
    StripePayout,
    StripePayoutError,
    StripeTransactionStatus,
    create_stripe_payout,
    get_payment_methods,
    get_stripe_balances,
    get_stripe_transaction_status,
)
from ypl.backend.user.user import get_user_vendor_profile
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrument,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
    PaymentTransaction,
    PaymentTransactionStatusEnum,
)
from ypl.db.point_transactions import PointsActionEnum
from ypl.db.users import VendorNameEnum

SYSTEM_USER_ID = "SYSTEM"
SLACK_WEBHOOK_CASHOUT = os.getenv("SLACK_WEBHOOK_CASHOUT", "")
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 1
RETRY_WAIT_MIN = 4
RETRY_WAIT_MAX = 15
STRIPE_PENDING_IDENTIFIER = "stripe-pending-account-creation"


class StripeFacilitator(BaseFacilitator):
    def __init__(
        self,
        currency: CurrencyEnum,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        facilitator: PaymentInstrumentFacilitatorEnum,
    ):
        super().__init__(currency, destination_identifier_type, facilitator)

    async def get_balance(self, currency: CurrencyEnum, payment_transaction_id: uuid.UUID | None = None) -> Decimal:
        """Get the balance for a specific currency.

        Args:
            currency: The currency to get balance for
            payment_transaction_id: The ID of the payment transaction, if this request is part of a payment transaction.
        Returns:
            Decimal: The balance amount
        """
        balances = await get_stripe_balances()
        balance = next((balance for balance in balances if balance.currency == self.currency.value), None)
        if not balance:
            raise PaymentInstrumentError(f"No balance found for currency: {self.currency.value}")
        return Decimal(balance.balance_amount)

    async def get_source_instrument(self) -> PaymentInstrument | None:
        return await get_generic_source_instrument(
            PaymentInstrumentFacilitatorEnum.STRIPE,
            PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
        )

    async def get_destination_instrument(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        account_id: str,
        instrument_metadata: dict | None = None,
    ) -> PaymentInstrument | None:
        if destination_identifier == STRIPE_PENDING_IDENTIFIER:
            payment_methods = await get_payment_methods(account_id)
            if not payment_methods:
                raise PaymentInstrumentError("No payment methods found for user")

            # For the moment, we only support the first payment method
            # TODO: Support multiple payment methods
            destination_identifier = payment_methods[0].id

            first_method = payment_methods[0]
            instrument_metadata = {
                "payment_method_id": destination_identifier,
                "object": first_method.object,
                "eligibility": first_method.eligibility,
                "eligibility_reason": first_method.eligibility_reason.__dict__,
                "created": first_method.created,
                "type": first_method.type.value,
            }

            # Add bank account details if present
            if first_method.bank_account:
                instrument_metadata["bank_account"] = {
                    "archived": first_method.bank_account.archived,
                    "bank_name": first_method.bank_account.bank_name,
                    "country": first_method.bank_account.country,
                    "last4": first_method.bank_account.last4,
                    "enabled_methods": first_method.bank_account.enabled_methods,
                    "supported_currencies": first_method.bank_account.supported_currencies,
                    "type": first_method.bank_account.type,
                }

            # Add card details if present
            if first_method.card:
                instrument_metadata["card"] = {
                    "archived": first_method.card.archived,
                    "exp_month": first_method.card.exp_month,
                    "exp_year": first_method.card.exp_year,
                    "last4": first_method.card.last4,
                    "type": first_method.card.type,
                }

        return await get_generic_destination_instrument(
            PaymentInstrumentFacilitatorEnum.STRIPE,
            user_id,
            destination_identifier,
            destination_identifier_type,
            instrument_metadata,
        )

    @staticmethod
    def map_stripe_status_to_internal(status: str) -> PaymentTransactionStatusEnum:
        """Map Stripe's transaction status to our internal PaymentTransactionStatusEnum.

        Args:
            status: The Stripe transaction status

        Returns:
            PaymentTransactionStatusEnum: The corresponding internal status
        """
        if status == StripeTransactionStatus.POSTED:
            return PaymentTransactionStatusEnum.SUCCESS
        elif status in (
            StripeTransactionStatus.FAILED,
            StripeTransactionStatus.RETURNED,
            StripeTransactionStatus.CANCELLED,
        ):
            return PaymentTransactionStatusEnum.FAILED
        else:
            return PaymentTransactionStatusEnum.PENDING

    async def _send_payment_request(
        self,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        usd_amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
        payment_transaction_id: uuid.UUID | None = None,
    ) -> PaymentResponse:
        start_time = time.time()

        try:
            try:
                credits_to_cashout += CASHOUT_TXN_COST

                # get balance to ensure enough funds
                balance = await self.get_balance(self.currency)
                if balance < amount:
                    raise PaymentInstrumentError("Insufficient balance")

                # get the recipient account details on stripe
                user_vendor_profile = await get_user_vendor_profile(user_id, VendorNameEnum.STRIPE)
                if not user_vendor_profile:
                    raise PaymentInstrumentError("User vendor profile not found")
                recipient_account_id = user_vendor_profile.user_vendor_id

                source_instrument = await self.get_source_instrument()
                source_instrument_id = source_instrument.payment_instrument_id if source_instrument else None
                destination_instrument = await self.get_destination_instrument(
                    user_id, destination_identifier, destination_identifier_type, recipient_account_id
                )
                destination_instrument_id = (
                    destination_instrument.payment_instrument_id if destination_instrument else None
                )

                if not source_instrument_id or not destination_instrument_id:
                    raise PaymentInstrumentError("Failed to get payment instruments")
            except Exception as e:
                log_dict = {
                    "message": "Failed to get payment instruments",
                    "user_id": user_id,
                    "error": str(e),
                    "destination_identifier": destination_identifier,
                }
                logging.exception(json_dumps(log_dict))
                raise PaymentInstrumentError("Failed to get payment instruments") from e

            try:
                payment_transaction_request = PaymentTransactionRequest(
                    currency=self.currency,
                    amount=amount,
                    usd_amount=usd_amount,
                    source_instrument_id=source_instrument_id,
                    destination_instrument_id=destination_instrument_id,
                    status=PaymentTransactionStatusEnum.NOT_STARTED,
                    additional_info={
                        "user_id": user_id,
                        "destination_identifier": destination_identifier,
                        "destination_identifier_type": str(destination_identifier_type),
                    },
                )
                payment_transaction_id = await create_payment_transaction(payment_transaction_request)
            except Exception as e:
                log_dict = {
                    "message": "Stripe: Failed to create payment transaction",
                    "user_id": user_id,
                    "amount": str(amount),
                    "usd_amount": str(usd_amount),
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                raise PaymentProcessingError("Stripe: Failed to create payment transaction") from e

            try:
                point_transaction_request = CashoutPointTransactionRequest(
                    user_id=user_id,
                    point_delta=-credits_to_cashout,
                    action_type=PointsActionEnum.CASHOUT,
                    cashout_payment_transaction_id=payment_transaction_id,
                )
                point_transaction_id = await create_cashout_point_transaction(point_transaction_request)
            except Exception as e:
                log_dict = {
                    "message": "Stripe: Failed to create point transaction",
                    "user_id": user_id,
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                await handle_failed_transaction(
                    payment_transaction_id,
                    None,
                    user_id,
                    credits_to_cashout,
                    amount,
                    usd_amount,
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=False,
                    currency=self.currency,
                )
                raise PaymentProcessingError("Stripe: Failed to create point transaction") from e

            try:
                await update_user_points(user_id, -credits_to_cashout)
            except Exception as e:
                log_dict = {
                    "message": "Stripe: Failed to update user points",
                    "user_id": user_id,
                    "payment_transaction_id": str(payment_transaction_id),
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                await handle_failed_transaction(
                    payment_transaction_id,
                    point_transaction_id,
                    user_id,
                    credits_to_cashout,
                    amount,
                    usd_amount,
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=False,
                    currency=self.currency,
                )
                raise PaymentProcessingError("Stripe: Failed to update user points") from e

            try:
                if not source_instrument or not destination_instrument:
                    raise PaymentInstrumentError("Failed to get payment instruments")
                stripe_payout = StripePayout(
                    from_account_id=source_instrument.identifier,
                    currency=self.currency.value,
                    recipient_account_id=recipient_account_id,
                    destination_id=destination_instrument.identifier,
                    amount=amount,
                )
                payout_id, payout_status, receipt_url = await create_stripe_payout(stripe_payout)

                await update_payment_transaction(
                    payment_transaction_id,
                    partner_reference_id=payout_id,
                    status=self.map_stripe_status_to_internal(payout_status),
                    customer_reference_id=receipt_url,
                )

                if payout_status not in (StripeTransactionStatus.POSTED):
                    # Start monitoring in background task
                    asyncio.create_task(
                        self._monitor_transaction_completion(
                            payout_id=payout_id,
                            payment_transaction_id=payment_transaction_id,
                            points_transaction_id=point_transaction_id,
                            user_id=user_id,
                            credits_to_cashout=credits_to_cashout,
                            amount=amount,
                            usd_amount=usd_amount,
                            source_instrument_id=source_instrument_id,
                            destination_instrument_id=destination_instrument_id,
                            destination_identifier=destination_identifier,
                            destination_identifier_type=destination_identifier_type,
                        )
                    )

                # Log success
                end_time = time.time()
                log_dict = {
                    "message": ":white_check_mark: Success - Stripe payout submitted",
                    "duration": str(end_time - start_time),
                    "user_id": user_id,
                    "amount": str(amount),
                    "credits_to_cashout": str(credits_to_cashout),
                    "source_instrument_id": str(source_instrument_id),
                    "destination_instrument_id": str(destination_instrument_id),
                    "destination_identifier": destination_identifier,
                    "currency": self.currency.value,
                    "payout_id": payout_id,
                    "payment_transaction_id": str(payment_transaction_id),
                    "points_transaction_id": str(point_transaction_id),
                    "receipt_url": receipt_url,
                }
                logging.info(json_dumps(log_dict))
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
                return PaymentResponse(
                    payment_transaction_id=payment_transaction_id,
                    transaction_status=PaymentTransactionStatusEnum.PENDING,
                    partner_reference_id=payout_id,
                    customer_reference_id=receipt_url,
                )

            except StripePayoutError as e:
                log_dict = {
                    "message": "Stripe: Failed to process Stripe payout",
                    "user_id": user_id,
                    "amount": str(amount),
                    "destination_identifier": destination_identifier,
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                await handle_failed_transaction(
                    payment_transaction_id,
                    point_transaction_id,
                    user_id,
                    credits_to_cashout,
                    amount,
                    usd_amount,
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=True,
                    currency=self.currency,
                )
                raise ValueError(str(e)) from e
            except Exception as e:
                log_dict = {
                    "message": "Stripe: Failed to process Stripe payout. Reversing transaction.",
                    "user_id": user_id,
                    "amount": str(amount),
                    "destination_identifier": destination_identifier,
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                await handle_failed_transaction(
                    payment_transaction_id,
                    point_transaction_id,
                    user_id,
                    credits_to_cashout,
                    amount,
                    usd_amount,
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=True,
                    currency=self.currency,
                )
                raise PaymentProcessingError("Stripe: Failed to process Stripe payout") from e

        except Exception as e:
            log_dict = {
                "message": "Stripe: Unexpected error in Stripe payout processing",
                "user_id": user_id,
                "amount": str(amount),
                "destination_identifier": destination_identifier,
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentProcessingError("Stripe: Payout processing failed") from e

    async def get_payment_status(self, payment_transaction_id: uuid.UUID) -> PaymentResponse:
        log_dict = {
            "message": "Getting Stripe payout status",
            "payment_transaction_id": str(payment_transaction_id),
            "currency": self.currency.value,
        }
        logging.info(json_dumps(log_dict))
        async with get_async_session() as session:
            query = select(PaymentTransaction).where(
                PaymentTransaction.payment_transaction_id == payment_transaction_id
            )
            result = await session.execute(query)
            transaction = result.scalar_one_or_none()
            if not transaction:
                raise PaymentStatusFetchError(f"Payment transaction {payment_transaction_id} not found")

            if transaction.status in (
                PaymentTransactionStatusEnum.SUCCESS,
                PaymentTransactionStatusEnum.FAILED,
                PaymentTransactionStatusEnum.REVERSED,
            ):
                return PaymentResponse(
                    payment_transaction_id=payment_transaction_id,
                    transaction_status=transaction.status,
                    customer_reference_id=str(transaction.customer_reference_id),
                    partner_reference_id=str(transaction.partner_reference_id),
                )

        status = await get_stripe_transaction_status(transaction.partner_reference_id)
        transaction_status = self.map_stripe_status_to_internal(status)

        return PaymentResponse(
            payment_transaction_id=payment_transaction_id,
            transaction_status=transaction_status,
            customer_reference_id=str(transaction.customer_reference_id),
            partner_reference_id=str(transaction.partner_reference_id),
        )

    async def _monitor_transaction_completion(
        self,
        payout_id: str,
        payment_transaction_id: uuid.UUID,
        points_transaction_id: uuid.UUID,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        usd_amount: Decimal,
        source_instrument_id: uuid.UUID,
        destination_instrument_id: uuid.UUID,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> None:
        """Monitor Stripe payout completion and handle success/failure.

        Args:
            payout_id: The Stripe payout ID to monitor
            payment_transaction_id: The ID of the payment transaction
            points_transaction_id: The ID of the points transaction
            user_id: The user ID
            credits_to_cashout: The number of credits being cashed out
            amount: The amount being transferred
            source_instrument_id: The source payment instrument ID
            destination_instrument_id: The destination payment instrument ID
            destination_identifier: The destination identifier
            destination_identifier_type: The type of destination identifier
        """
        try:
            start_time = time.time()
            max_wait_time, poll_interval = self.get_polling_config()

            # For Stripe, we rely on webhooks to update the transaction status
            # This monitoring is just a backup in case the webhook fails
            while (time.time() - start_time) < max_wait_time:
                stripe_status = await get_stripe_transaction_status(payout_id)
                if stripe_status in (
                    StripeTransactionStatus.CANCELLED,
                    StripeTransactionStatus.FAILED,
                    StripeTransactionStatus.RETURNED,
                ):
                    log_dict = {
                        "message": "Stripe transaction failed",
                        "user_id": user_id,
                        "transaction_id": payout_id,
                        "status": stripe_status,
                        "elapsed_time": time.time() - start_time,
                    }
                    logging.error(json_dumps(log_dict))

                    # Handle the failed transaction
                    await handle_failed_transaction(
                        payment_transaction_id=payment_transaction_id,
                        points_transaction_id=points_transaction_id,
                        user_id=user_id,
                        credits_to_cashout=credits_to_cashout,
                        amount=amount,
                        usd_amount=usd_amount,
                        source_instrument_id=source_instrument_id,
                        destination_instrument_id=destination_instrument_id,
                        destination_identifier=destination_identifier,
                        destination_identifier_type=destination_identifier_type,
                        update_points=True,
                        currency=self.currency,
                    )
                    return

                if stripe_status in (StripeTransactionStatus.POSTED):
                    await update_payment_transaction(
                        payment_transaction_id,
                        partner_reference_id=payout_id,
                        status=PaymentTransactionStatusEnum.SUCCESS,
                        customer_reference_id=payout_id,
                    )
                    log_dict = {
                        "message": "Stripe transaction posted",
                        "user_id": user_id,
                        "transaction_id": payout_id,
                        "status": stripe_status,
                        "elapsed_time": time.time() - start_time,
                    }
                    logging.info(json_dumps(log_dict))
                    return

                await asyncio.sleep(poll_interval)

            # If we get here, we've timed out
            # Do not reverse the transaction here as the webhook might still come through
            log_dict = {
                "message": ":x: Failure - Stripe payout monitoring timed out\n"
                f"payout_id: {payout_id}\n"
                f"payment_transaction_id: {payment_transaction_id}\n"
                f"points_transaction_id: {points_transaction_id}\n"
                f"user_id: {user_id}\n"
                f"credits_to_cashout: {credits_to_cashout}\n"
                f"amount: {amount}\n"
                f"usd_amount: {usd_amount}\n"
                f"source_instrument_id: {source_instrument_id}\n"
                f"destination_instrument_id: {destination_instrument_id}\n"
                f"destination_identifier: {destination_identifier}\n"
                f"destination_identifier_type: {destination_identifier_type}\n",
            }
            logging.error(json_dumps(log_dict))

            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

        except Exception as e:
            log_dict = {
                "message": ":x: Failure - Error monitoring Stripe payout completion",
                "user_id": user_id,
                "payout_id": payout_id,
                "error": str(e),
                "elapsed_time": str(time.time() - start_time),
            }
            logging.error(json_dumps(log_dict))
            raise PaymentProcessingError("Failed to monitor Stripe payout completion") from e

    @staticmethod
    def get_polling_config() -> tuple[int, int]:
        """
        Returns polling configuration for Stripe transactions.
        Returns:
            tuple: (max_wait_time_seconds, poll_interval_seconds)
        """
        return (
            5 * 60,  # 5 minutes in seconds
            30,  # 30 seconds
        )

    async def get_source_instrument_id(self) -> UUID:
        source_instrument = await self.get_source_instrument()
        if not source_instrument:
            raise PaymentInstrumentError("Failed to get source instrument")
        return source_instrument.payment_instrument_id

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        instrument_metadata: dict | None = None,
    ) -> UUID:
        user_vendor_profile = await get_user_vendor_profile(user_id, VendorNameEnum.STRIPE)
        if not user_vendor_profile:
            raise PaymentInstrumentError("User vendor profile not found")

        destination_instrument = await self.get_destination_instrument(
            user_id,
            destination_identifier,
            destination_identifier_type,
            user_vendor_profile.user_vendor_id,
            instrument_metadata,
        )
        if not destination_instrument:
            raise PaymentInstrumentError("Failed to get destination instrument")
        return destination_instrument.payment_instrument_id
