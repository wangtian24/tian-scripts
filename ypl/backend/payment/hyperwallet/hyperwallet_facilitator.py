import asyncio
import logging
import os
import time
import uuid
from decimal import Decimal

import httpx
from fastapi import HTTPException
from sqlalchemy import select
from tenacity import retry, stop_after_attempt, wait_exponential
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.payment.facilitator import (
    BaseFacilitator,
    PaymentInstrumentError,
    PaymentProcessingError,
    PaymentResponse,
    PaymentStatusFetchError,
    PointTransactionCreationError,
    TransactionCreationError,
)
from ypl.backend.payment.hyperwallet.hyperwallet_payout import (
    HyperwalletPayout,
    HyperwalletPayoutError,
    TransactionStatus,
    get_transaction_status,
    process_hyperwallet_payout,
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
from ypl.backend.user.user import RegisterVendorRequest, register_user_with_vendor
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
from ypl.db.users import UserVendorProfile, VendorNameEnum

SYSTEM_USER_ID = "SYSTEM"
SLACK_WEBHOOK_CASHOUT = os.getenv("SLACK_WEBHOOK_CASHOUT")
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 1
RETRY_WAIT_MIN = 4
RETRY_WAIT_MAX = 15


class HyperwalletFacilitator(BaseFacilitator):
    def __init__(
        self,
        currency: CurrencyEnum,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        facilitator: PaymentInstrumentFacilitatorEnum,
    ):
        super().__init__(currency, destination_identifier_type, facilitator)

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    )
    async def get_source_instrument_id(self) -> uuid.UUID:
        return await get_generic_source_instrument_id(
            PaymentInstrumentFacilitatorEnum.HYPERWALLET,
            PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
        )

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        instrument_metadata: dict | None = None,
    ) -> uuid.UUID:
        return await get_destination_instrument_id(
            PaymentInstrumentFacilitatorEnum.HYPERWALLET,
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

    async def create_transfer_method(
        self,
        user_token: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        transfer_method_country: str = "US",
        transfer_method_currency: str = "USD",
    ) -> str:
        """Create a transfer method for a user.

        Args:
            user_token: The Hyperwallet user token
            email: The email address for the transfer method
            transfer_method_country: The country code for the transfer method (default: US)
            transfer_method_currency: The currency code for the transfer method (default: USD)

        Returns:
            str: The transfer method token

        Raises:
            PaymentInstrumentError: If the transfer method creation fails
        """
        try:
            api_username = settings.hyperwallet_username
            api_password = settings.hyperwallet_password
            api_url = settings.hyperwallet_api_url

            if not all([api_username, api_password, api_url]):
                raise PaymentInstrumentError("Hyperwallet: Missing API credentials")

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            auth = (api_username, api_password)

            if destination_identifier_type == PaymentInstrumentIdentifierTypeEnum.PAYPAL_ID:
                account_type = "PAYPAL_ACCOUNT"
                payload = {
                    "type": account_type,
                    "transferMethodCountry": transfer_method_country,
                    "transferMethodCurrency": transfer_method_currency,
                    "email": destination_identifier,
                }
                url = f"{api_url}/users/{user_token}/paypal-accounts"
            elif destination_identifier_type == PaymentInstrumentIdentifierTypeEnum.VENMO_ID:
                account_type = "VENMO_ACCOUNT"
                payload = {
                    "type": account_type,
                    "transferMethodCountry": transfer_method_country,
                    "transferMethodCurrency": transfer_method_currency,
                    "accountId": destination_identifier,
                }
                url = f"{api_url}/users/{user_token}/venmo-accounts"
            else:
                raise PaymentInstrumentError("Hyperwallet: Unsupported transfer method type")

            log_dict = {
                "message": f"Hyperwallet: Creating {account_type} transfer method",
                "user_token": user_token,
                "payload": payload,
            }
            logging.info(json_dumps(log_dict))

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    auth=auth,
                )
                response.raise_for_status()

                data = response.json()
                transfer_token = data.get("token")

                if not transfer_token:
                    log_dict = {
                        "message": "Hyperwallet: No transfer token in response",
                        "user_token": user_token,
                        "response": str(data),
                    }
                    logging.error(json_dumps(log_dict))
                    raise PaymentInstrumentError("Hyperwallet: No transfer token in response")

                log_dict = {
                    "message": f"Hyperwallet: Created {account_type} transfer method",
                    "user_token": user_token,
                    "transfer_token": transfer_token,
                }
                logging.info(json_dumps(log_dict))

                return str(transfer_token)

        except Exception as e:
            log_dict = {
                "message": f"Hyperwallet: Failed to create {account_type} transfer method",
                "user_token": user_token,
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentInstrumentError(f"Hyperwallet: Failed to create {account_type} transfer method") from e

    async def get_payment_metadata(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
    ) -> dict:
        try:
            #  Hyperwallet needs a user token. See if the user is already registered
            async with get_async_session() as session:
                query = select(UserVendorProfile).where(
                    UserVendorProfile.user_id == user_id,  # type: ignore
                    UserVendorProfile.vendor_name == VendorNameEnum.HYPERWALLET,  # type: ignore
                    UserVendorProfile.deleted_at.is_(None),  # type: ignore
                )
                result = await session.execute(query)
                user_vendor = result.scalar_one_or_none()
                if not user_vendor:
                    log_dict = {
                        "message": "Hyperwallet: User token not found",
                        "user_id": user_id,
                    }
                    logging.info(json_dumps(log_dict))
                    #  if user is not registered, register the user
                    user_vendor = await register_user_with_vendor(
                        RegisterVendorRequest(
                            user_id=user_id,
                            vendor_name=VendorNameEnum.HYPERWALLET.value,
                            additional_details=destination_additional_details,
                        )
                    )
                    user_token = user_vendor.user_vendor_id
                else:
                    user_token = user_vendor.user_vendor_id

                log_dict = {
                    "message": "Hyperwallet: User token found",
                    "user_id": user_id,
                    "user_vendor_id": user_vendor.user_vendor_id,
                }
                logging.info(json_dumps(log_dict))

                # see if the transfer method already exists
                async with get_async_session() as session:
                    query = select(PaymentInstrument).where(
                        PaymentInstrument.user_id == user_id,  # type: ignore
                        PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.HYPERWALLET,  # type: ignore
                        PaymentInstrument.identifier_type == destination_identifier_type,  # type: ignore
                        PaymentInstrument.identifier == destination_identifier,  # type: ignore
                        PaymentInstrument.deleted_at.is_(None),  # type: ignore
                    )
                    result = await session.execute(query)
                    payment_instrument = result.scalar_one_or_none()
                    if (
                        payment_instrument
                        and payment_instrument.instrument_metadata
                        and payment_instrument.instrument_metadata.get("transfer_token")
                    ):
                        transfer_token = payment_instrument.instrument_metadata["transfer_token"]
                    else:
                        transfer_token = await self.create_transfer_method(
                            user_token=user_token,
                            destination_identifier=destination_identifier,
                            destination_identifier_type=destination_identifier_type,
                        )
                instrument_metadata = {
                    "user_token": user_token,
                    "transfer_token": transfer_token,
                }

                return instrument_metadata
        except Exception as e:
            log_dict = {
                "message": "Hyperwallet: Failed to get payment token",
                "user_id": user_id,
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentInstrumentError("Hyperwallet: Failed to get payment token") from e

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
                # TODO: Implement this as the documentation doesn't tell how to get the balance
                # Get the balance of the source instrument
                # source_instrument_balance = await self.get_balance(self.currency, payment_transaction_id)

                # if source_instrument_balance < amount:
                #     log_dict = {
                #         "message": "Source instrument does not have enough balance",
                #         "user_id": user_id,
                #         "amount": str(amount),
                #         "source_instrument_balance": str(source_instrument_balance),
                #     }
                #     logging.error(json_dumps(log_dict))
                #     raise PaymentInstrumentError("Source instrument does not have enough balance")

                payment_metadata = await self.get_payment_metadata(
                    user_id, destination_identifier, destination_identifier_type, destination_additional_details
                )
                source_instrument_id = await self.get_source_instrument_id()
                destination_instrument_id = await self.get_destination_instrument_id(
                    user_id, destination_identifier, destination_identifier_type, payment_metadata
                )
            except HTTPException as e:
                raise e
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
                    "message": "Hyperwallet: Failed to create payment transaction",
                    "user_id": user_id,
                    "amount": str(amount),
                    "usd_amount": str(usd_amount),
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                raise TransactionCreationError("Hyperwallet: Failed to create payment transaction") from e

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
                    "message": "Hyperwallet: Failed to create point transaction",
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
                raise PointTransactionCreationError("Hyperwallet: Failed to create point transaction") from e

            try:
                await update_user_points(user_id, -credits_to_cashout)
            except Exception as e:
                log_dict = {
                    "message": "Hyperwallet: Failed to update user points",
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
                raise PaymentProcessingError("Hyperwallet: Failed to update user points") from e

            try:
                hyperwallet_payout = HyperwalletPayout(
                    user_id=user_id,
                    amount=amount,
                    currency=self.currency,
                    payment_transaction_id=payment_transaction_id,
                    destination_token=payment_metadata["transfer_token"],
                )
                transaction_token, transaction_status = await process_hyperwallet_payout(hyperwallet_payout)

                await update_payment_transaction(
                    payment_transaction_id,
                    partner_reference_id=transaction_token,
                    status=self._map_hyperwallet_status_to_internal(transaction_status),
                    customer_reference_id=transaction_token,
                )

                if transaction_status not in (TransactionStatus.COMPLETED.value, TransactionStatus.FAILED.value):
                    # Start monitoring in background task
                    asyncio.create_task(
                        self._monitor_transaction_completion(
                            transaction_token=transaction_token,
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
                    "message": ":white_check_mark: Success - Hyperwallet payout submitted",
                    "duration": str(end_time - start_time),
                    "user_id": user_id,
                    "amount": str(amount),
                    "credits_to_cashout": str(credits_to_cashout),
                    "source_instrument_id": str(source_instrument_id),
                    "destination_instrument_id": str(destination_instrument_id),
                    "destination_identifier": destination_identifier,
                    "currency": self.currency.value,
                    "transaction_token": transaction_token,
                    "payment_transaction_id": str(payment_transaction_id),
                    "points_transaction_id": str(point_transaction_id),
                }
                logging.info(json_dumps(log_dict))
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
                return PaymentResponse(
                    payment_transaction_id=payment_transaction_id,
                    transaction_status=PaymentTransactionStatusEnum.PENDING,
                    customer_reference_id=transaction_token,
                )

            except HyperwalletPayoutError as e:
                log_dict = {
                    "message": "Hyperwallet: Failed to process Hyperwallet payout",
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
                    "message": "Hyperwallet: Failed to process Hyperwallet payout. Reversing transaction.",
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
                raise PaymentProcessingError("Hyperwallet: Failed to process Hyperwallet payout") from e

        except HTTPException as e:
            raise e
        except Exception as e:
            log_dict = {
                "message": "Hyperwallet: Unexpected error in Hyperwallet payout processing",
                "user_id": user_id,
                "amount": str(amount),
                "destination_identifier": destination_identifier,
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentProcessingError("Hyperwallet: Payout processing failed") from e

    @staticmethod
    def _map_hyperwallet_status_to_internal(status: str) -> PaymentTransactionStatusEnum:
        """Map Hyperwallet's transaction status to our internal PaymentTransactionStatusEnum.

        Args:
            status: The Hyperwallet transaction status

        Returns:
            PaymentTransactionStatusEnum: The corresponding internal status
        """
        if status == TransactionStatus.COMPLETED.value:
            return PaymentTransactionStatusEnum.SUCCESS
        elif status in (
            TransactionStatus.FAILED.value,
            TransactionStatus.CANCELLED.value,
            TransactionStatus.RECALLED.value,
            TransactionStatus.RETURNED.value,
            TransactionStatus.EXPIRED.value,
        ):
            return PaymentTransactionStatusEnum.FAILED
        elif status in (
            TransactionStatus.CREATED.value,
            TransactionStatus.SCHEDULED.value,
            TransactionStatus.PENDING_ACCOUNT_ACTIVATION.value,
            TransactionStatus.PENDING_TAX_VERIFICATION.value,
            TransactionStatus.PENDING_TRANSFER_METHOD_ACTION.value,
            TransactionStatus.PENDING_TRANSACTION_VERIFICATION.value,
            TransactionStatus.IN_PROGRESS.value,
            TransactionStatus.UNCLAIMED.value,
        ):
            return PaymentTransactionStatusEnum.PENDING
        else:
            return PaymentTransactionStatusEnum.PENDING

    async def get_payment_status(self, payment_transaction_id: uuid.UUID) -> PaymentResponse:
        log_dict = {
            "message": "Getting Hyperwallet payout status",
            "payment_transaction_id": str(payment_transaction_id),
            "currency": self.currency.value,
        }
        async with get_async_session() as session:
            query = select(PaymentTransaction).where(
                PaymentTransaction.payment_transaction_id == payment_transaction_id  # type: ignore
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
        status = await get_transaction_status(partner_reference_id)
        log_dict = {
            "message": "Hyperwallet: Retrieved transaction status",
            "payment_transaction_id": str(payment_transaction_id),
            "status": status,
        }
        logging.info(json_dumps(log_dict))
        return PaymentResponse(
            payment_transaction_id=payment_transaction_id,
            transaction_status=self._map_hyperwallet_status_to_internal(status),
            customer_reference_id=str(transaction.customer_reference_id),
        )

    async def _monitor_transaction_completion(
        self,
        transaction_token: str,
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
        """Monitor Hyperwallet payout completion and handle success/failure.

        Args:
            transaction_token: The Hyperwallet transaction token to monitor
            payment_transaction_id: The ID of the payment transaction
            points_transaction_id: The ID of the points transaction
            user_id: The user ID
            credits_to_cashout: The number of credits being cashed out
            amount: The amount being transferred
            source_instrument_id: The source payment instrument ID
            destination_instrument_id: The destination payment instrument ID
            destination_identifier: The destination token
            destination_identifier_type: The type of destination identifier
        """
        try:
            start_time = time.time()
            max_wait_time, poll_interval = self.get_polling_config()

            while (time.time() - start_time) < max_wait_time:
                status = await get_transaction_status(transaction_token)
                if status == TransactionStatus.COMPLETED.value:
                    await update_payment_transaction(
                        payment_transaction_id,
                        partner_reference_id=transaction_token,
                        status=PaymentTransactionStatusEnum.SUCCESS,
                        customer_reference_id=transaction_token,
                    )
                    log_dict = {
                        "message": "Hyperwallet payout completed",
                        "user_id": user_id,
                        "transaction_token": transaction_token,
                        "status": status,
                        "elapsed_time": time.time() - start_time,
                    }
                    logging.info(json_dumps(log_dict))
                    return
                elif status in (
                    TransactionStatus.FAILED.value,
                    TransactionStatus.CANCELLED.value,
                    TransactionStatus.EXPIRED.value,
                    TransactionStatus.RECALLED.value,
                    TransactionStatus.RETURNED.value,
                ):
                    log_dict = {
                        "message": "Hyperwallet payout failed",
                        "user_id": user_id,
                        "transaction_token": transaction_token,
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
                "message": ":x: Failure - Hyperwallet payout monitoring timed out\n"
                f"transaction_token: {transaction_token}\n"
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

            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

        except Exception as e:
            log_dict = {
                "message": ":x: Failure - Error monitoring Hyperwallet payout completion",
                "user_id": user_id,
                "transaction_token": transaction_token,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }
            logging.error(json_dumps(log_dict))
            raise PaymentProcessingError("Failed to monitor Hyperwallet payout completion") from e

    @staticmethod
    def get_polling_config() -> tuple[int, int]:
        """
        Returns polling configuration for Hyperwallet transactions.
        Returns:
            tuple: (max_wait_time_seconds, poll_interval_seconds)
        """
        return (
            1 * 60,  # 1 minutes in seconds
            10,  # 10 seconds
        )
