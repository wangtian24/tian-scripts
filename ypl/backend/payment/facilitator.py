import asyncio
import logging
import os
import time
import uuid
from decimal import Decimal

from cdp.transaction import Transaction
from cdp.transfer import Transfer
from tenacity import retry, stop_after_attempt, wait_exponential
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.payment.base_types import (
    BaseFacilitator,
    PaymentInstrumentError,
    PaymentProcessingError,
    PaymentResponse,
    PointTransactionCreationError,
    TransactionCreationError,
)
from ypl.backend.payment.crypto.crypto_payout import CryptoReward, get_crypto_balance, process_single_crypto_reward
from ypl.backend.payment.payment import (
    CashoutPointTransactionRequest,
    PaymentTransactionRequest,
    create_cashout_point_transaction,
    create_payment_transaction,
    update_payment_transaction,
    update_user_points,
)
from ypl.backend.payment.payout_utils import CASHOUT_TXN_COST, handle_failed_transaction
from ypl.backend.payment.payout_utils import (
    get_destination_instrument_id as get_generic_destination_instrument_id,
)
from ypl.backend.payment.payout_utils import (
    get_source_instrument_id as get_generic_source_instrument_id,
)
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import fetch_user_name
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
    PaymentTransactionStatusEnum,
)
from ypl.db.point_transactions import PointsActionEnum

SYSTEM_USER_ID = "SYSTEM"
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 1
RETRY_WAIT_MIN = 4
RETRY_WAIT_MAX = 15
SLACK_WEBHOOK_CASHOUT = os.getenv("SLACK_WEBHOOK_CASHOUT")


class UpiFacilitator(BaseFacilitator):
    async def get_balance(self, currency: CurrencyEnum) -> Decimal:
        # TODO: Implement this
        return Decimal(1000)

    async def get_source_instrument_id(self) -> uuid.UUID:
        # TODO: Implement this
        return uuid.uuid4()

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> uuid.UUID:
        # TODO: Implement this
        return uuid.uuid4()

    async def _send_payment_request(
        self,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
    ) -> PaymentResponse:
        # TODO: Implement this
        return PaymentResponse(
            payment_transaction_id=uuid.uuid4(),
            transaction_status=PaymentTransactionStatusEnum.SUCCESS,
            customer_reference_id="1234567890",
        )

    async def get_payment_status(self, payment_reference_id: str) -> PaymentTransactionStatusEnum:
        # TODO: Implement this
        return PaymentTransactionStatusEnum.SUCCESS


class OnChainFacilitator(BaseFacilitator):
    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    )
    async def get_source_instrument_id(self) -> uuid.UUID:
        return await get_generic_source_instrument_id(
            PaymentInstrumentFacilitatorEnum.ON_CHAIN,
            PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
        )

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> uuid.UUID:
        return await get_generic_destination_instrument_id(
            PaymentInstrumentFacilitatorEnum.ON_CHAIN,
            user_id,
            destination_identifier,
            destination_identifier_type,
        )

    async def get_balance(self, currency: CurrencyEnum) -> Decimal:
        crypto_balance = await get_crypto_balance(currency)
        return crypto_balance

    async def _send_payment_request(
        self,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
    ) -> PaymentResponse:
        # Flow of the function
        # 0. Get the balance of the source instrument
        # 1. Get the source and destination instrument ids
        # 2. Create a payment transaction
        # 3. Create a point transaction entry for the cashout
        # 4. Update the user points
        # 5. Process the crypto reward
        # 6. Monitor the transfer completion
        # 7. If success then update the payment transaction status
        # 8. If failure then reverse the payment transaction, point transaction and update the user points
        start_time = time.time()
        try:
            try:
                credits_to_cashout += CASHOUT_TXN_COST
                # 0. Get the balance of the source instrument
                source_instrument_balance = await self.get_balance(self.currency)

                if source_instrument_balance < amount:
                    log_dict = {
                        "message": "Source instrument does not have enough balance",
                        "user_id": user_id,
                        "amount": str(amount),
                        "source_instrument_balance": str(source_instrument_balance),
                    }
                    logging.error(json_dumps(log_dict))
                    raise ValueError("Source instrument does not have enough balance")

                source_instrument_id = await self.get_source_instrument_id()
                destination_instrument_id = await self.get_destination_instrument_id(
                    user_id, destination_identifier, destination_identifier_type
                )
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
                    source_instrument_id=source_instrument_id,
                    destination_instrument_id=destination_instrument_id,
                    status=PaymentTransactionStatusEnum.NOT_STARTED,
                    additional_info={
                        "user_id": user_id,
                        "destination_identifier": destination_identifier,
                        "destination_identifier_type": destination_identifier_type,
                    },
                )
                payment_transaction_id = await create_payment_transaction(payment_transaction_request)
            except Exception as e:
                log_dict = {
                    "message": "Failed to create payment transaction",
                    "user_id": user_id,
                    "amount": str(amount),
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                raise TransactionCreationError("Failed to create payment transaction") from e

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
                    "message": "Failed to create point transaction",
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
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=False,
                    currency=self.currency,
                )
                raise PointTransactionCreationError("Failed to create point transaction") from e

            try:
                await update_user_points(user_id, -credits_to_cashout)
            except Exception as e:
                log_dict = {
                    "message": "Failed to update user points or transaction status",
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
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=False,
                    currency=self.currency,
                )
                raise PaymentProcessingError("Failed to update user points or transaction status") from e

            try:
                tx_hash, transfer = await process_single_crypto_reward(
                    CryptoReward(
                        user_id=user_id,
                        wallet_address=destination_identifier,
                        asset_id=self.currency.value.lower(),
                        amount=amount,
                    )
                )
                await update_payment_transaction(
                    payment_transaction_id,
                    partner_reference_id=tx_hash,
                    status=self._map_transaction_status_to_internal(str(transfer.status).lower()),
                )

                if str(transfer.status).lower() != Transaction.Status.COMPLETE.value.lower():
                    # Start monitoring in background task only if not complete
                    asyncio.create_task(
                        self._monitor_transfer_completion(
                            transfer=transfer,
                            payment_transaction_id=payment_transaction_id,
                            points_transaction_id=point_transaction_id,
                            user_id=user_id,
                            credits_to_cashout=credits_to_cashout,
                            amount=amount,
                            source_instrument_id=source_instrument_id,
                            destination_instrument_id=destination_instrument_id,
                            destination_identifier=destination_identifier,
                            destination_identifier_type=destination_identifier_type,
                        )
                    )

                # Log success
                end_time = time.time()
                log_dict = {
                    "message": "Successfully submitted for crypto cashout",
                    "duration": str(end_time - start_time),
                    "user_id": user_id,
                    "amount": str(amount),
                    "credits_to_cashout": str(credits_to_cashout),
                    "source_instrument_id": str(source_instrument_id),
                    "destination_instrument_id": str(destination_instrument_id),
                    "destination_identifier": destination_identifier,
                    "currency": self.currency.value,
                    "tx_hash": str(tx_hash),
                    "transfer_id": transfer.transfer_id,
                }
                logging.info(json_dumps(log_dict))
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
                return PaymentResponse(
                    payment_transaction_id=payment_transaction_id,
                    transaction_status=PaymentTransactionStatusEnum.PENDING,
                    customer_reference_id=tx_hash,
                )

            except Exception as e:
                log_dict = {
                    "message": "Failed to process crypto reward. Reversing transaction.",
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
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=True,
                    currency=self.currency,
                )
                raise PaymentProcessingError("Failed to process crypto reward") from e

        except Exception as e:
            log_dict = {
                "message": "Unexpected error in payment processing",
                "user_id": user_id,
                "amount": str(amount),
                "destination_identifier": destination_identifier,
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentProcessingError("Payment processing failed") from e

    async def get_payment_status(self, payment_reference_id: str) -> PaymentTransactionStatusEnum:
        # TODO: Implement this
        return PaymentTransactionStatusEnum.SUCCESS

    async def _monitor_transfer_completion(
        self,
        transfer: Transfer,
        payment_transaction_id: uuid.UUID,
        points_transaction_id: uuid.UUID,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        source_instrument_id: uuid.UUID,
        destination_instrument_id: uuid.UUID,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> None:
        """Monitor transfer completion and handle success/failure.

        Args:
            transfer: The transfer to monitor
            payment_transaction_id: The ID of the payment transaction
            user_id: The user ID
            credits_to_cashout: The number of credits being cashed out
            amount: The amount being transferred
            source_instrument_id: The source payment instrument ID
            destination_instrument_id: The destination payment instrument ID
            destination_identifier: The destination wallet address
            destination_identifier_type: The type of destination identifier
        """
        try:
            start_time = time.time()
            max_wait_time = 300  # 5 minutes in seconds
            poll_interval = 5  # Check every 5 seconds

            # first check if the transfer is already complete
            # if not then wait for coinbase designed wait
            if str(transfer.status).lower() != Transaction.Status.COMPLETE.value.lower():
                try:
                    transfer.wait()
                except Exception as e:
                    log_dict = {
                        "message": "Error waiting for transfer completion",
                        "user_id": user_id,
                        "transaction_hash": transfer.transaction_hash,
                        "transfer_id": transfer.transfer_id,
                        "error": str(e),
                        "elapsed_time": time.time() - start_time,
                    }
                    logging.error(json_dumps(log_dict))

            # if still not complete then wait for max wait time
            while (
                str(transfer.status).lower() != Transaction.Status.COMPLETE.value.lower()
                and (time.time() - start_time) < max_wait_time
            ):
                transfer.reload()
                await asyncio.sleep(poll_interval)

            if str(transfer.status).lower() == Transaction.Status.COMPLETE.value.lower():
                await update_payment_transaction(
                    payment_transaction_id,
                    partner_reference_id=transfer.transaction_hash,
                    status=PaymentTransactionStatusEnum.SUCCESS,
                )
                log_dict = {
                    "message": "Crypto transfer completed",
                    "user_id": user_id,
                    "transaction_hash": transfer.transaction_hash,
                    "status": transfer.status,
                    "transfer_id": transfer.transfer_id,
                    "elapsed_time": time.time() - start_time,
                }
                logging.info(json_dumps(log_dict))
            elif str(transfer.status).lower() == Transaction.Status.FAILED.value.lower():
                user_name = await fetch_user_name(user_id)
                log_dict = {
                    "message": ":red_circle: *Crypto transfer failed*",
                    "transaction_id": transfer.transaction_hash,
                    "payment_transaction_id": payment_transaction_id,
                    "points_transaction_id": points_transaction_id,
                    "user_id": user_id,
                    "user_name": user_name,
                    "credits_to_cashout": credits_to_cashout,
                    "amount": amount,
                    "source_instrument_id": source_instrument_id,
                    "destination_instrument_id": destination_instrument_id,
                    "destination_identifier": destination_identifier,
                    "destination_identifier_type": destination_identifier_type,
                    "status": transfer.status,
                    "transfer_id": transfer.transfer_id,
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
                    source_instrument_id=source_instrument_id,
                    destination_instrument_id=destination_instrument_id,
                    destination_identifier=destination_identifier,
                    destination_identifier_type=destination_identifier_type,
                    update_points=True,
                    currency=self.currency,
                )
            else:
                # TODO: Send alert to Slack
                # do not reverse the transaction here as the txn might still complete
                log_dict = {
                    "message": ":red_circle: *Crypto transfer monitoring timed out*",
                    "transaction_id": transfer.transaction_hash,
                    "payment_transaction_id": payment_transaction_id,
                    "points_transaction_id": points_transaction_id,
                    "user_id": user_id,
                    "credits_to_cashout": credits_to_cashout,
                    "amount": amount,
                    "source_instrument_id": source_instrument_id,
                    "destination_instrument_id": destination_instrument_id,
                    "destination_identifier": destination_identifier,
                    "destination_identifier_type": destination_identifier_type,
                    "status": transfer.status,
                    "transfer_id": transfer.transfer_id,
                    "elapsed_time": time.time() - start_time,
                }
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

        except Exception as e:
            log_dict = {
                "message": "Error monitoring transfer completion",
                "user_id": user_id,
                "transaction_hash": transfer.transaction_hash,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }
            logging.error(json_dumps(log_dict))
            raise PaymentProcessingError("Failed to monitor transfer completion") from e

    @staticmethod
    def _map_transaction_status_to_internal(status: str) -> PaymentTransactionStatusEnum:
        """Map Transaction.Status to our internal PaymentTransactionStatusEnum.

        Args:
            status: The transaction status from Transaction.Status

        Returns:
            PaymentTransactionStatusEnum: The corresponding internal status
        """
        if status == Transaction.Status.COMPLETE.value:
            return PaymentTransactionStatusEnum.SUCCESS
        elif status == Transaction.Status.FAILED.value:
            return PaymentTransactionStatusEnum.FAILED
        elif status == Transaction.Status.PENDING.value:
            return PaymentTransactionStatusEnum.PENDING
        else:
            # For unknown status, keep it as PENDING since we don't know if it failed
            return PaymentTransactionStatusEnum.PENDING
