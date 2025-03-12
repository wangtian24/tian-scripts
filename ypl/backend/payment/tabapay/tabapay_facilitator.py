import asyncio
import logging
import os
import time
import uuid
from dataclasses import asdict
from decimal import Decimal

from sqlmodel import select
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name_bg
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
    get_destination_instrument_id,
    handle_failed_transaction,
)
from ypl.backend.payment.payout_utils import (
    get_source_instrument_id as get_generic_source_instrument_id,
)
from ypl.backend.payment.tabapay.tabapay_payout import TabaPayClient
from ypl.backend.utils.async_utils import create_background_task
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
from ypl.partner_payments.server.partner.tabapay.client import (
    TabapayAccountCreationRequest,
    TabapayAchEntryTypeEnum,
    TabapayAchOptionsEnum,
    TabapayBankInfo,
    TabapayCardCreationInfo,
    TabapayOwnerInfo,
    TabapayOwnerName,
    TabapayOwnerPhone,
    TabapayStatusEnum,
    TabapayTransactionAccounts,
    TabapayTransactionRequest,
)

SYSTEM_USER_ID = "SYSTEM"
SLACK_WEBHOOK_CASHOUT = os.getenv("SLACK_WEBHOOK_CASHOUT", "")
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 1
RETRY_WAIT_MIN = 4
RETRY_WAIT_MAX = 15
TABAPAY_PENDING_IDENTIFIER = "tabapay-pending-account-creation"


class TabaPayFacilitator(BaseFacilitator):
    def __init__(
        self,
        currency: CurrencyEnum,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        facilitator: PaymentInstrumentFacilitatorEnum,
    ):
        super().__init__(currency, destination_identifier_type, facilitator)
        self.client = TabaPayClient()

    async def cleanup(self) -> None:
        """Clean up resources used by the facilitator."""
        if self.client:
            await self.client.cleanup()

    async def get_source_instrument_id(self) -> uuid.UUID:
        return await get_generic_source_instrument_id(
            PaymentInstrumentFacilitatorEnum.TABAPAY,
            PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
        )

    async def check_and_create_destination_instrument(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
    ) -> PaymentInstrument:
        #  check if the instrument already exists or is this a new account creation
        if destination_identifier != TABAPAY_PENDING_IDENTIFIER:
            async with get_async_session() as session:
                query = select(PaymentInstrument).where(
                    PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.TABAPAY,
                    PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
                    PaymentInstrument.identifier == destination_identifier,
                    PaymentInstrument.deleted_at.is_(None),  # type: ignore
                )
                result = await session.exec(query)
                existing_instrument = result.one_or_none()
                if existing_instrument:
                    return existing_instrument

        #  if the instrument does not exist, we need to create it
        #  we need to call the create account API and pass the destination_additional_details
        #  we need to save and return the payment instrument
        if destination_additional_details is None:
            raise PaymentProcessingError(
                "TabaPay: Destination additional details are required for new account creation"
            )

        try:
            account_details = TabapayAccountCreationRequest(
                referenceID=str(user_id)[:15],
                owner=TabapayOwnerInfo(
                    name=TabapayOwnerName(
                        first=destination_additional_details["first_name"],
                        last=destination_additional_details["last_name"],
                    ),
                    phone=TabapayOwnerPhone(
                        number=destination_additional_details["phone_number"],
                        countryCode=destination_additional_details["phone_country_code"],
                    ),
                ),
                bank=(
                    TabapayBankInfo(
                        routingNumber=destination_additional_details["routing_number"],
                        accountNumber=destination_additional_details["account_number"],
                        accountType=destination_additional_details["account_type"],
                    )
                    if "routing_number" in destination_additional_details
                    else None
                )
                if "card_token" not in destination_additional_details
                else None,
                card=(
                    TabapayCardCreationInfo(
                        token=destination_additional_details["card_token"].split("|", 2)[-1]
                        if destination_additional_details["card_token"]
                        else "",
                    )
                    if "card_token" in destination_additional_details
                    else None
                ),
            )
            async with self.client as client:
                account_response = await client.create_account(account_details)
                account_id = account_response.account_id
                account_metadata = account_response.metadata

            payment_instrument = PaymentInstrument(
                facilitator=PaymentInstrumentFacilitatorEnum.TABAPAY,
                identifier_type=PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
                identifier=account_id,
                user_id=user_id,
                instrument_metadata=account_metadata,
            )
            async with get_async_session() as session:
                session.add(payment_instrument)
                await session.commit()

            return payment_instrument

        except Exception as e:
            raise PaymentProcessingError("TabaPay: Failed to create account") from e

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        instrument_metadata: dict | None = None,
    ) -> uuid.UUID:
        return await get_destination_instrument_id(
            PaymentInstrumentFacilitatorEnum.TABAPAY,
            user_id,
            destination_identifier,
            destination_identifier_type,
            instrument_metadata,
        )

    async def get_balance(self, currency: CurrencyEnum, payment_transaction_id: uuid.UUID | None = None) -> Decimal:
        """Get the balance for a specific currency.

        Args:
            currency: The currency to get balance for
            payment_transaction_id: The ID of the payment transaction, if this request is part of a payment transaction.
        Returns:
            Decimal: The balance amount
        """
        # TODO: Implement this
        return Decimal(0)

    @staticmethod
    def _map_tabapay_status_to_internal(status: TabapayStatusEnum) -> PaymentTransactionStatusEnum:
        """Map TabaPay's transaction status to our internal PaymentTransactionStatusEnum.

        Args:
            status: The TabaPay transaction status

        Returns:
            PaymentTransactionStatusEnum: The corresponding internal status
        """
        if status == TabapayStatusEnum.COMPLETED:
            return PaymentTransactionStatusEnum.SUCCESS
        elif status in (
            TabapayStatusEnum.FAILED,
            TabapayStatusEnum.ERROR,
            TabapayStatusEnum.REVERSAL,
            TabapayStatusEnum.REVERSED,
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
                source_instrument_id = await self.get_source_instrument_id()
                destination_instrument_details = await self.check_and_create_destination_instrument(
                    user_id, destination_identifier, destination_identifier_type, destination_additional_details
                )
                destination_instrument_id = destination_instrument_details.payment_instrument_id
                destination_identifier = destination_instrument_details.identifier
                destination_metadata = destination_instrument_details.instrument_metadata

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
                    "message": "TabaPay: Failed to create payment transaction",
                    "user_id": user_id,
                    "amount": str(amount),
                    "usd_amount": str(usd_amount),
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                raise PaymentProcessingError("TabaPay: Failed to create payment transaction") from e

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
                    "message": "TabaPay: Failed to create point transaction",
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
                raise PaymentProcessingError("TabaPay: Failed to create point transaction") from e

            try:
                await update_user_points(user_id, -credits_to_cashout)
            except Exception as e:
                log_dict = {
                    "message": "TabaPay: Failed to update user points",
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
                raise PaymentProcessingError("TabaPay: Failed to update user points") from e

            try:
                # Create TabaPay transaction request
                # purpose of payment is required for cards and varies different for different networks
                # check if the destination is bank and then get the rtp details
                ach_options = None
                ach_entry_type = None

                if destination_metadata and destination_metadata.get("bank"):
                    ach_entry_type = TabapayAchEntryTypeEnum.WEB
                    async with self.client as client:
                        rtp = await client.get_rtp_details(destination_metadata["routing_number"])
                    if rtp:
                        ach_options = TabapayAchOptionsEnum.RTP
                    else:
                        ach_options = TabapayAchOptionsEnum.NEXT_DAY

                tabapay_request = TabapayTransactionRequest(
                    referenceID=str(payment_transaction_id)[:15],
                    accounts=TabapayTransactionAccounts(
                        sourceAccountID=str(source_instrument_id),
                        destinationAccountID=destination_identifier,
                    ),
                    currency=self.currency.value,
                    amount=amount,
                    achEntryType=ach_entry_type,
                    achOptions=ach_options,
                )

                async with self.client as client:
                    transaction_response = await client.process_payout(tabapay_request)

                    if not transaction_response:  # incase none was returned due to some exception
                        transaction_id = str(payment_transaction_id)[:15]
                        status = TabapayStatusEnum.PENDING
                    elif transaction_response.networkRC == "00":
                        transaction_id = transaction_response.transactionID
                        status = TabapayStatusEnum.COMPLETED
                    else:
                        transaction_id = transaction_response.transactionID
                        status = transaction_response.status

                await update_payment_transaction(
                    payment_transaction_id,
                    partner_reference_id=transaction_id,
                    status=self._map_tabapay_status_to_internal(TabapayStatusEnum(status)),
                    customer_reference_id=transaction_id,
                    additional_info={
                        "transaction_response": asdict(transaction_response) if transaction_response else None,
                    },
                )

                if status not in (TabapayStatusEnum.COMPLETED, TabapayStatusEnum.FAILED, TabapayStatusEnum.ERROR):
                    # Start monitoring in background task
                    create_background_task(
                        self._monitor_transaction_completion(
                            transaction_id=transaction_id,
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
                    "message": ":white_check_mark: Success - TabaPay transaction submitted",
                    "duration": str(end_time - start_time),
                    "user_id": user_id,
                    "amount": str(amount),
                    "credits_to_cashout": str(credits_to_cashout),
                    "source_instrument_id": str(source_instrument_id),
                    "destination_instrument_id": str(destination_instrument_id),
                    "destination_identifier": destination_identifier,
                    "currency": self.currency.value,
                    "transaction_id": transaction_id,
                    "payment_transaction_id": str(payment_transaction_id),
                    "points_transaction_id": str(point_transaction_id),
                }
                logging.info(json_dumps(log_dict))
                post_to_slack_with_user_name_bg(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT)
                return PaymentResponse(
                    payment_transaction_id=payment_transaction_id,
                    transaction_status=PaymentTransactionStatusEnum.PENDING,
                    customer_reference_id=transaction_id,
                )

            except Exception as e:
                log_dict = {
                    "message": "TabaPay: Failed to process TabaPay transaction",
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
                raise PaymentProcessingError("TabaPay: Failed to process TabaPay transaction") from e

        except Exception as e:
            log_dict = {
                "message": "TabaPay: Unexpected error in TabaPay transaction processing",
                "user_id": user_id,
                "amount": str(amount),
                "destination_identifier": destination_identifier,
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentProcessingError("TabaPay: Transaction processing failed") from e

    async def get_payment_status(self, payment_transaction_id: uuid.UUID) -> PaymentResponse:
        log_dict = {
            "message": "Getting TabaPay transaction status",
            "payment_transaction_id": str(payment_transaction_id),
            "currency": self.currency.value,
        }
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
                    customer_reference_id=str(transaction.partner_reference_id),
                    partner_reference_id=str(transaction.partner_reference_id),
                )

        partner_reference_id = str(transaction.partner_reference_id)
        status = await self.client.get_transaction_status(partner_reference_id)
        log_dict = {
            "message": "TabaPay: Retrieved transaction status",
            "payment_transaction_id": str(payment_transaction_id),
            "status": status,
        }
        logging.info(json_dumps(log_dict))
        return PaymentResponse(
            payment_transaction_id=payment_transaction_id,
            transaction_status=self._map_tabapay_status_to_internal(TabapayStatusEnum(status)),
            customer_reference_id=str(transaction.customer_reference_id),
        )

    async def _monitor_transaction_completion(
        self,
        transaction_id: str,
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
        """Monitor TabaPay transaction completion and handle success/failure.

        Args:
            transaction_id: The TabaPay transaction ID to monitor
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

            while (time.time() - start_time) < max_wait_time:
                status = await self.client.get_transaction_status(transaction_id)
                if status == TabapayStatusEnum.COMPLETED:
                    await update_payment_transaction(
                        payment_transaction_id,
                        partner_reference_id=transaction_id,
                        status=PaymentTransactionStatusEnum.SUCCESS,
                        customer_reference_id=transaction_id,
                    )
                    log_dict = {
                        "message": "TabaPay transaction completed",
                        "user_id": user_id,
                        "transaction_id": transaction_id,
                        "status": status,
                        "elapsed_time": time.time() - start_time,
                    }
                    logging.info(json_dumps(log_dict))
                    return
                elif status in (TabapayStatusEnum.FAILED, TabapayStatusEnum.ERROR):
                    log_dict = {
                        "message": "TabaPay transaction failed",
                        "user_id": user_id,
                        "transaction_id": transaction_id,
                        "status": status,
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

                await asyncio.sleep(poll_interval)

            # If we get here, we've timed out
            # Do not reverse the transaction here as the txn might still complete
            log_dict = {
                "message": ":x: Failure - TabaPay transaction monitoring timed out\n"
                f"transaction_id: {transaction_id}\n"
                f"payment_transaction_id: {payment_transaction_id}\n"
                f"points_transaction_id: {points_transaction_id}\n"
                f"user_id: {user_id}\n"
                f"credits_to_cashout: {credits_to_cashout}\n"
                f"amount: {amount}\n"
                f"usd_amount: {usd_amount}\n"
                f"source_instrument_id: {source_instrument_id}\n"
                f"destination_instrument_id: {destination_instrument_id}\n"
                f"destination_identifier: {destination_identifier}\n"
                f"destination_identifier_type: {destination_identifier_type}\n"
                f"status: {status}\n",
            }
            logging.error(json_dumps(log_dict))

            post_to_slack_with_user_name_bg(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT)

        except Exception as e:
            log_dict = {
                "message": ":x: Failure - Error monitoring TabaPay transaction completion",
                "user_id": user_id,
                "transaction_id": transaction_id,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }
            logging.error(json_dumps(log_dict))
            raise PaymentProcessingError("Failed to monitor TabaPay transaction completion") from e

    @staticmethod
    def get_polling_config() -> tuple[int, int]:
        """
        Returns polling configuration for TabaPay transactions.
        Returns:
            tuple: (max_wait_time_seconds, poll_interval_seconds)
        """
        return (
            1 * 60,  # 1 minutes in seconds
            10,  # 10 seconds
        )
