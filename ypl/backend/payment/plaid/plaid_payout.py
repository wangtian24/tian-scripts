import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any

import plaid
from plaid.model.transfer_authorization_decision import TransferAuthorizationDecision
from ypl.backend.payment.plaid_lifecycle import get_plaid_client
from ypl.backend.utils.json import json_dumps


class AccountType(str, Enum):
    """Account types supported by Plaid."""

    CHECKING = "checking"
    SAVINGS = "savings"


@dataclass
class PlaidPayout:
    user_id: str
    user_name: str
    amount: Decimal
    account_number: str
    routing_number: str
    account_type: str


async def process_plaid_payout(payout: PlaidPayout) -> tuple[str, str]:
    """Process a Plaid payout.

    Returns:
        Tuple[str, str]: A tuple containing (transfer_id, transfer_status)
    """

    # log the request
    log_dict = {
        "message": "Processing Plaid payout",
        "user_id": payout.user_id,
        "user_name": payout.user_name,
        "amount": payout.amount,
        "account_number": payout.account_number,
        "routing_number": payout.routing_number,
        "account_type": payout.account_type,
    }
    logging.info(json_dumps(log_dict))

    #  if any of the input values are not populated, raise an error
    if (
        not payout.user_id
        or not payout.user_name
        or not payout.amount
        or not payout.account_number
        or not payout.routing_number
        or not payout.account_type
    ):
        log_dict = {
            "message": "Invalid input values for Plaid payout",
        }
        logging.error(json_dumps(log_dict))
        raise ValueError("Invalid input values for Plaid payout.")

    client = get_plaid_client()
    if client is None:
        raise ValueError("Plaid client not initialized.")

    available_balance = await get_balance()
    if available_balance < payout.amount:
        log_dict = {
            "message": "Insufficient balance to make payment",
            "user_id": payout.user_id,
            "available_balance": str(available_balance),
            "payout_amount": str(payout.amount),
        }
        logging.error(json_dumps(log_dict))
        raise ValueError("Insufficient balance to make payment")

    plaid_item = await create_plaid_item(
        payout.account_number,
        payout.routing_number,
        payout.account_type,
    )

    rtp_eligibility = await check_rtp_eligibility(plaid_item["access_token"], plaid_item["account_id"])
    network = "rtp" if rtp_eligibility else "ach"

    # Round the amount to 2 decimal places for currency
    # as plaid requires the amount to be in 2 digits only
    rounded_amount = payout.amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    transfer_authorization = await create_transfer_authorization(
        access_token=plaid_item["access_token"],
        account_id=plaid_item["account_id"],
        type="credit",
        network=network,
        amount=str(rounded_amount),
        user={"legal_name": payout.user_name},
    )

    authorization = transfer_authorization.get("authorization", {})
    authorization_id = authorization.get("id")
    authorization_decision = authorization.get("decision")

    log_dict = {
        "message": "Transfer authorization created",
        "user_id": payout.user_id,
        "amount": payout.amount,
        "transfer_authorization_id": str(authorization_id),
        "transfer_authorization_decision": str(authorization_decision),
    }
    logging.info(json_dumps(log_dict))

    if str(authorization_decision).lower() == TransferAuthorizationDecision.allowed_values[("value",)]["APPROVED"]:
        transfer_id, transfer_status = await create_transfer(
            access_token=plaid_item["access_token"],
            account_id=plaid_item["account_id"],
            authorization_id=authorization_id,
            description="payment",
        )
        log_dict = {
            "message": "Transfer created",
            "user_id": payout.user_id,
            "amount": payout.amount,
            "transfer_id": str(transfer_id),
            "transfer_status": str(transfer_status),
        }
        logging.info(json_dumps(log_dict))
    else:
        log_dict = {
            "message": "Transfer authorization failed",
            "user_id": payout.user_id,
            "amount": payout.amount,
            "transfer_authorization_id": str(authorization_id),
            "transfer_authorization_decision": str(authorization_decision),
        }
        logging.error(json_dumps(log_dict))
        raise ValueError(f"Transfer authorization failed: decision={authorization_decision}")

    return transfer_id, transfer_status


async def create_plaid_item(account_number: str, routing_number: str, account_type: str) -> dict[str, str]:
    """
    Create a Plaid item and retrieve access token, account ID and request ID.

    Args:
        account_number (str): The account number of the bank account
        routing_number (str): The routing number of the bank account
        account_type (AccountType): The type of account (default: AccountType.CHECKING)

    Returns:
        dict: Dictionary containing access_token, account_id, and request_id
    """
    try:
        client = get_plaid_client()
        request = {
            "account_number": account_number,
            "routing_number": routing_number,
            "account_type": account_type,
        }

        response = client.transfer_migrate_account(request)

        return {
            "access_token": response.access_token,
            "account_id": response.account_id,
            "request_id": response.request_id,
        }
    except plaid.ApiException as e:
        print(f"Error creating bank account linking token: {e}")
        raise


async def create_transfer_authorization(
    access_token: str,
    account_id: str,
    type: str,
    network: str,
    amount: str,
    user: dict[str, Any],
) -> dict[str, Any]:
    """
    Create a transfer authorization using Plaid's API.

    Args:
        access_token (str): The Plaid access token
        account_id (str): The Plaid account ID
        type (str): Type of transfer (e.g., "credit", "debit")
        network (str): Payment network (e.g., "ach", "rtp")
        amount (str): The amount to transfer
        user (dict): User information dictionary containing legal_name

    Returns:
        dict[str, Any]: Dictionary containing transfer authorization details
    """
    try:
        client = get_plaid_client()
        request = {
            "access_token": access_token,
            "account_id": account_id,
            "type": type,
            "network": network,
            "amount": amount,
            "user": user,
        }

        if type == "ach":
            request["ach_class"] = "ccd"

        response = client.transfer_authorization_create(request)
        if not response or not response.authorization:
            raise plaid.ApiException("Invalid response from Plaid")

        response_dict: dict[str, Any] = response.to_dict()
        return response_dict
    except plaid.ApiException as e:
        print(f"Error creating transfer authorization: {e}")
        raise


async def create_transfer(
    access_token: str, account_id: str, authorization_id: str, description: str = "payment"
) -> tuple[str, str]:
    """
    Create a transfer using Plaid's API.

    Args:
        access_token (str): The Plaid access token
        account_id (str): The Plaid account ID
        authorization_id (str): The ID from the transfer authorization
        description (str): Description of the transfer (default: "payment")

    Returns:
        Tuple[str, str]: A tuple containing (transfer_id, transfer_status)
    """
    try:
        client = get_plaid_client()
        request = {
            "access_token": access_token,
            "account_id": account_id,
            "authorization_id": authorization_id,
            "description": description,
        }

        response = client.transfer_create(request)
        if not response or not response.transfer:
            raise plaid.ApiException("Invalid response from Plaid")

        return response.transfer.id, response.transfer.status
    except plaid.ApiException as e:
        log_dict = {
            "message": "Error creating transfer",
            "access_token": access_token,
            "account_id": account_id,
            "authorization_id": authorization_id,
            "description": description,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise


async def get_balance() -> Decimal:
    """
    Get the ledger balance information including available and pending balances.

    Returns:
        Decimal: The available balance
    """
    try:
        client = get_plaid_client()
        response = client.transfer_ledger_get({})
        if not response or not response.balance:
            raise plaid.ApiException("Invalid response from Plaid")

        return Decimal(str(response.balance.available))
    except plaid.ApiException as e:
        log_dict = {"message": "Error getting ledger balance", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise


async def cancel_transfer(transfer_id: str) -> dict[str, Any]:
    """
    Cancel a transfer using Plaid's API.

    Args:
        transfer_id (str): The ID of the transfer to cancel

    Returns:
        dict[str, Any]: Plaid response containing the cancelled transfer details
    """
    try:
        client = get_plaid_client()
        request = {"transfer_id": transfer_id}

        response = client.transfer_cancel(request)
        if not response:
            raise plaid.ApiException("Invalid response from Plaid")

        response_dict: dict[str, Any] = response.to_dict()
        return response_dict
    except plaid.ApiException as e:
        log_dict = {"message": "Error cancelling transfer", "transfer_id": str(transfer_id), "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise


async def fund_sandbox_account() -> None:
    """
    Fund the sandbox account by cancelling all pending transfers.
    """
    try:
        pending_transfers = await get_pending_transfers()
        print("Found pending transfers:", pending_transfers)

        # Cancel all pending transfers
        for transfer in pending_transfers["transfers"]:
            print(f"Cancelling transfer {transfer['id']}")
            await cancel_transfer(transfer["id"])

        balance = await get_balance()
        print("New balance: ", balance)

    except plaid.ApiException as e:
        log_dict = {"message": "Error funding sandbox account", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise


async def get_pending_transfers(
    start_date: str | None = None, end_date: str | None = None, count: int = 20, offset: int = 0
) -> dict[str, Any]:
    """
    Get the pending transfers within a date range.

    Args:
        start_date (Optional[str]): Start date in ISO format (e.g., '2019-12-06T22:35:49Z')
        end_date (Optional[str]): End date in ISO format (e.g., '2019-12-12T22:35:49Z')
        count (int): Number of transfers to return (default: 20)
        offset (int): Number of transfers to skip (default: 0)

    Returns:
        dict[str, Any]: Plaid response containing list of pending transfers
    """
    try:
        client = get_plaid_client()
        request: dict[str, Any] = {"count": count, "offset": offset}

        if start_date:
            request["start_date"] = start_date
        if end_date:
            request["end_date"] = end_date

        response = client.transfer_list(request)
        if not response or not response.transfers:
            raise plaid.ApiException("Invalid response from Plaid")

        response_dict: dict[str, Any] = response.to_dict()
        return response_dict
    except plaid.ApiException as e:
        log_dict = {"message": "Error getting transfers", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise


async def check_rtp_eligibility(access_token: str, account_id: str) -> bool:
    """
    Check if an account is eligible for RTP (Real-Time Payments) transfers.

    Args:
        access_token (str): The Plaid access token
        account_id (str): The Plaid account ID

    Returns:
        bool: True if RTP credit transfers are supported, False otherwise
    """
    try:
        client = get_plaid_client()
        request = {"access_token": access_token, "account_id": account_id}

        response = client.transfer_capabilities_get(request)
        if not response:
            raise plaid.ApiException("Invalid response from Plaid")

        response_dict = response.to_dict()
        is_rtp_credit_supported = (
            response_dict.get("institution_supported_networks", {}).get("rtp", {}).get("credit", False)
        )

        return bool(is_rtp_credit_supported)

    except plaid.ApiException as e:
        log_dict = {
            "message": "Error checking RTP eligibility",
            "access_token": access_token,
            "account_id": account_id,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise


async def get_plaid_transfer_status(transfer_id: str) -> str:
    """
    Get the status and details of a specific transfer.

    Args:
        transfer_id (str): The ID of the transfer to check

    Returns:
        str: The status of the transfer
    """
    try:
        client = get_plaid_client()
        request = {"transfer_id": transfer_id}

        response = client.transfer_get(request)
        if not response or not response.transfer:
            raise plaid.ApiException("Invalid response from Plaid")

        return str(response.transfer.status)
    except plaid.ApiException as e:
        log_dict = {"message": "Error getting transfer status", "transfer_id": transfer_id, "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise
