import logging
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from functools import wraps
from typing import Any, TypeVar, cast

import httpx
from ypl.backend.utils.json import json_dumps
from ypl.partner_payments.server.common.types import GetBalanceRequest, GetBalanceResponse
from ypl.partner_payments.server.config import secret_manager
from ypl.partner_payments.server.partner.base import BasePartnerClient

CONTENT_TYPE_JSON = "application/json"
AUTH_BEARER = "Bearer"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3

F = TypeVar("F", bound=Callable[..., Any])


class TabapayStatusEnum(str, Enum):
    """Represents all possible Tabapay resource and transaction statuses."""

    PENDING = "PENDING"
    BATCH = "BATCH"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"
    COMPLETED = "COMPLETED"
    REVERSED = "REVERSED"
    REVERSAL = "REVERSAL"


class TabapayAccountTypeEnum(str, Enum):
    """Account types for Tabapay transactions."""

    C = "CHECKING"
    S = "SAVINGS"


class TabapayAchOptionsEnum(str, Enum):
    """ACH Options for Tabapay transactions."""

    R = "RTP"
    N = "Next Day"
    S = "Same Day"


class TabapayAchEntryTypeEnum(str, Enum):
    """ACH Entry Types for Tabapay transactions."""

    PPD = "PPD"  # Prearranged payment and deposit
    CCD = "CCD"  # Commercial credit and deposit
    WEB = "WEB"  # Internet payment


@dataclass(frozen=True)
class TabapayConfig:
    """Configuration for Tabapay client.

    Attributes:
        api_url: Base URL for Tabapay API
        client_id: Client identifier for authentication
        bearer_token: Key for authentication
    """

    api_url: str
    client_id: str
    bearer_token: str

    def validate(self) -> None:
        """Validates that all required fields are present."""
        if not self.api_url:
            raise TabapayError("Tabapay API URL is missing")
        if not self.client_id:
            raise TabapayError("Tabapay client ID is missing")
        if not self.bearer_token:
            raise TabapayError("Tabapay bearer token is missing")


@dataclass(frozen=True)
class TabapayBankInfo:
    """Bank account information for Tabapay transactions."""

    routingNumber: str
    accountNumber: str
    accountType: TabapayAccountTypeEnum


@dataclass(frozen=True)
class TabapayCardInfo:
    """Card information for Tabapay transactions."""

    last4: str
    expirationDate: str


@dataclass(frozen=True)
class TabapayOwnerName:
    """Owner name information."""

    first: str
    last: str


@dataclass(frozen=True)
class TabapayOwnerAddress:
    """Owner address information."""

    line1: str
    line2: str
    city: str
    state: str
    zipcode: str


@dataclass(frozen=True)
class TabapayOwnerPhone:
    """Owner phone information."""

    number: str


@dataclass(frozen=True)
class TabapayOwnerInfo:
    """Complete owner information including name, phone, and optional address."""

    name: TabapayOwnerName
    phone: TabapayOwnerPhone
    address: TabapayOwnerAddress | None = None


@dataclass(frozen=True)
class TabapayAccountDetails:
    """Complete account details including owner info and optional bank/card details."""

    owner: TabapayOwnerInfo
    bank: TabapayBankInfo | None = None
    card: TabapayCardInfo | None = None


@dataclass(frozen=True)
class TabapayTransactionFees:
    """Fee structure for a Tabapay transaction."""

    interchange: Decimal
    network: Decimal
    tabapay: Decimal


@dataclass(frozen=True)
class TabapayTransactionDetails:
    """Details of a Tabapay transaction."""

    network: str
    networkRC: str
    status: TabapayStatusEnum
    approvalCode: str
    amount: Decimal
    fees: TabapayTransactionFees


@dataclass(frozen=True)
class TabapayTransactionAccounts:
    """Account IDs involved in a Tabapay transaction."""

    sourceAccountID: str
    destinationAccountID: str


@dataclass(frozen=True)
class TabapayTransactionRequest:
    """Request object for creating a Tabapay transaction.

    Attributes:
        referenceID: Unique identifier for the transaction. We will used payment_transaction_id
        accounts: Source and destination account information
        currency: Transaction currency code
        amount: Transaction amount
        purposeOfPayment: Description of payment purpose
        memo: Additional transaction notes
        type: Transaction type (default: "push")
        achOptions: Optional ACH-specific configuration
        achEntryType: Required for ACH transactions
    """

    referenceID: str
    accounts: TabapayTransactionAccounts
    currency: str
    amount: Decimal
    type: str = "push"
    memo: str | None = None
    purposeOfPayment: str | None = None
    achOptions: TabapayAchOptionsEnum | None = None
    achEntryType: TabapayAchEntryTypeEnum | None = None


@dataclass(frozen=True)
class TabapayCreateAccountResponse:
    """Response from Tabapay account creation.

    Attributes:
        account_id: The ID of the created account
        metadata: Additional data returned by Tabapay (card or bank info)
    """

    account_id: str
    metadata: dict[str, Any]


class TabapayError(Exception):
    """Base exception class for Tabapay related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

        # Log the error and post to Slack
        log_dict = {"message": f"Tabapay Error: {message}", **self.details}
        logging.error(json_dumps(log_dict))


def require_initialization(func: F) -> F:
    """Decorator to ensure client is initialized before method execution."""

    @wraps(func)
    async def wrapper(self: "TabaPayClient", *args: Any, **kwargs: Any) -> Any:
        if not self.http_client or not self._config_obj:
            await self.initialize()
            assert self.http_client is not None and self._config_obj is not None
        return await func(self, *args, **kwargs)

    return cast(F, wrapper)


class TabaPayClient(BasePartnerClient):
    """Client for interacting with the Tabapay API.

    This client handles all interactions with Tabapay's API including account management,
    transaction processing, and balance inquiries. It implements proper initialization,
    cleanup, and error handling patterns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.http_client: httpx.AsyncClient | None = None
        self._config_obj: TabapayConfig | None = None

    async def initialize(self) -> None:
        """Initialize the client with configuration and HTTP client setup."""
        if self.http_client is not None:
            return

        self.config = await secret_manager.get_tabapay_config()
        config_obj = TabapayConfig(
            api_url=self.config["api_url"], client_id=self.config["client_id"], bearer_token=self.config["bearer_token"]
        )
        config_obj.validate()

        self._config_obj = config_obj
        self.http_client = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT_SECONDS, limits=httpx.Limits(max_keepalive_connections=MAX_RETRIES)
        )

    async def cleanup(self) -> None:
        """Clean up resources used by the client. Should be called when the client is no longer needed."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
            self._config_obj = None

    def _get_headers(self) -> dict[str, str]:
        """Get common headers for API requests."""
        assert self._config_obj is not None
        return {"accept": CONTENT_TYPE_JSON, "authorization": f"{AUTH_BEARER} {self._config_obj.bearer_token}"}

    def _get_base_url(self, path: str) -> str:
        """Construct full URL for API endpoints."""
        assert self._config_obj is not None
        return f"{self._config_obj.api_url}/v1/clients/{self._config_obj.client_id}/{path}"

    @require_initialization
    async def get_balance(self, request: GetBalanceRequest) -> GetBalanceResponse:
        """Get the balance of the account from Tabapay.

        Args:
            request: Balance request parameters

        Returns:
            GetBalanceResponse containing the account balance

        Raises:
            TabapayError: If the balance request fails
        """
        # TODO: Implement actual balance fetching logic
        logging.info(json_dumps({"message": "fetching balance from tabapay"}))
        logging.info(json_dumps({"message": "fetched balance from tabapay"}))
        return GetBalanceResponse(balance=Decimal(1000), ip_address="unknown")

    @require_initialization
    async def get_rtp_details(self, routing_number: str) -> bool:
        """Get the RTP details from Tabapay.

        Args:
            routing_number: The routing number to check for RTP details

        Returns:
            bool: True if RTP details were successfully retrieved

        Raises:
            TabapayError: If the RTP details request fails
        """
        logging.info(json_dumps({"message": "fetching rtp details from tabapay", "routing_number": routing_number}))
        url = self._get_base_url("banks")
        assert self.http_client is not None

        try:
            response = await self.http_client.post(
                url, headers=self._get_headers(), json={"routingNumber": routing_number}
            )
            response.raise_for_status()
            data = response.json()
            logging.info(json_dumps({"message": "fetched rtp details from tabapay", "response": json_dumps(data)}))
            return bool(data.get("RTP", False))
        except httpx.HTTPError as e:
            error_details = {
                "routing_number": routing_number,
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None)
                if isinstance(e, httpx.HTTPStatusError)
                else None,
            }
            logging.warning(json_dumps(error_details))
            return False

    @require_initialization
    async def get_account_details(self, account_id: str) -> TabapayAccountDetails:
        """Get the details of the account from Tabapay.

        Args:
            account_id: The ID of the account to fetch

        Returns:
            TabapayAccountDetails containing the account information

        Raises:
            TabapayError: If the account details request fails
        """
        logging.info(json_dumps({"message": "fetching account details from tabapay", "account_id": account_id}))

        url = self._get_base_url(f"accounts/{account_id}")
        assert self.http_client is not None

        try:
            response = await self.http_client.get(url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()

            logging.info(
                json_dumps({"message": "fetched account details from tabapay", "status_code": response.status_code})
            )

            return self._parse_account_details(data)
        except httpx.HTTPError as e:
            error_details = {
                "account_id": account_id,
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None)
                if isinstance(e, httpx.HTTPStatusError)
                else None,
            }
            raise TabapayError("Error fetching account details from Tabapay", error_details) from e

    def _parse_account_details(self, data: dict[str, Any]) -> TabapayAccountDetails:
        """Parse raw API response into TabapayAccountDetails object."""
        owner_data = data.get("owner", {})
        address_data = owner_data.get("address", {})

        return TabapayAccountDetails(
            owner=TabapayOwnerInfo(
                name=TabapayOwnerName(first=owner_data.get("firstName", ""), last=owner_data.get("lastName", "")),
                address=TabapayOwnerAddress(
                    line1=address_data.get("line1", ""),
                    line2=address_data.get("line2", ""),
                    city=address_data.get("city", ""),
                    state=address_data.get("state", ""),
                    zipcode=address_data.get("zipcode", ""),
                )
                if address_data
                else None,
                phone=TabapayOwnerPhone(number=owner_data.get("phone", "")),
            ),
            bank=TabapayBankInfo(
                routingNumber=data.get("bank", {}).get("routingNumber", ""),
                accountNumber=data.get("bank", {}).get("last4", ""),
                accountType=TabapayAccountTypeEnum(data.get("bank", {}).get("accountType", "")),
            )
            if data.get("bank")
            else None,
            card=TabapayCardInfo(
                last4=data.get("card", {}).get("last4", ""),
                expirationDate=data.get("card", {}).get("expirationDate", ""),
            )
            if data.get("card")
            else None,
        )

    @require_initialization
    async def get_transaction_status(self, transaction_id: str) -> TabapayStatusEnum:
        """Get the status of a transaction from Tabapay.

        Args:
            transaction_id: The ID of the transaction to check

        Returns:
            TabapayStatusEnum indicating the current transaction status

        Raises:
            TabapayError: If the status request fails
        """
        logging.info(json_dumps({"message": "Tabapay: fetching transaction status", "transaction_id": transaction_id}))

        url = self._get_base_url(f"transactions/{transaction_id}")
        assert self.http_client is not None

        try:
            response = await self.http_client.get(url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()

            logging.info(
                json_dumps(
                    {
                        "message": "Tabapay: fetched transaction status",
                        "status_code": response.status_code,
                        "data": json_dumps(data),
                    }
                )
            )

            return TabapayStatusEnum(str(data.get("status", "")))
        except httpx.HTTPError as e:
            error_details = {
                "transaction_id": transaction_id,
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None)
                if isinstance(e, httpx.HTTPStatusError)
                else None,
            }
            raise TabapayError("Error fetching transaction details from Tabapay", error_details) from e

    @require_initialization
    async def create_account(self, request: TabapayAccountDetails) -> TabapayCreateAccountResponse:
        """Create an account in Tabapay.

        Args:
            request: Account details for creation

        Returns:
            TabapayCreateAccountResponse containing the account ID and additional metadata

        Raises:
            TabapayError: If account creation fails
        """
        log_dict = {
            "message": "Tabapay: Creating account",
            "request": {
                "owner": {
                    "name": {"first": request.owner.name.first, "last": request.owner.name.last},
                    "phone": {"number": request.owner.phone.number},
                    "address": request.owner.address.__dict__ if request.owner.address else None,
                },
                "bank": request.bank.__dict__ if request.bank else None,
                "card": request.card.__dict__ if request.card else None,
            },
        }
        logging.info(json_dumps(log_dict))

        url = self._get_base_url("accounts")
        assert self.http_client is not None

        try:
            response = await self.http_client.post(url, headers=self._get_headers(), json=request)
            response.raise_for_status()
            data = response.json()

            log_dict["message"] = "Tabapay: Account created"
            log_dict["response"] = json_dumps(data)
            logging.info(json_dumps(log_dict))

            # Extract metadata (everything except standard fields)
            metadata = {k: v for k, v in data.items() if k not in ["SC", "EC", "accountID"]}

            log_dict = {
                "message": "Tabapay: Account created",
                "response": json_dumps(data),
            }
            logging.info(json_dumps(log_dict))

            return TabapayCreateAccountResponse(
                account_id=str(data.get("accountID", "")),
                metadata=metadata,
            )
        except httpx.HTTPError as e:
            error_details = {
                "request": {
                    "owner": {
                        "name": {"first": request.owner.name.first, "last": request.owner.name.last},
                        "phone": {"number": request.owner.phone.number},
                        "address": request.owner.address.__dict__ if request.owner.address else None,
                    },
                    "bank": request.bank.__dict__ if request.bank else None,
                    "card": request.card.__dict__ if request.card else None,
                },
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None)
                if isinstance(e, httpx.HTTPStatusError)
                else None,
            }
            raise TabapayError("Error creating account in Tabapay", error_details) from e

    @require_initialization
    async def create_transaction(self, request: TabapayTransactionRequest) -> tuple[str, TabapayStatusEnum]:
        """Create a transaction in Tabapay.

        Args:
            request: Transaction details for creation

        Returns:
            tuple containing:
                - str: The ID of the created transaction
                - TabapayStatusEnum: The initial status of the transaction

        Raises:
            TabapayError: If transaction creation fails
        """
        log_dict = {
            "message": "Tabapay: Creating transaction",
            "request": {
                "referenceID": request.referenceID,
                "accounts": request.accounts.__dict__,
                "currency": request.currency,
                "amount": str(request.amount),
                "type": request.type,
                "memo": request.memo,
                "purposeOfPayment": request.purposeOfPayment,
                "achOptions": request.achOptions.value if request.achOptions else None,
                "achEntryType": request.achEntryType.value if request.achEntryType else None,
            },
        }
        logging.info(json_dumps(log_dict))

        url = self._get_base_url("transactions")
        assert self.http_client is not None

        try:
            response = await self.http_client.post(url, headers=self._get_headers(), json=request)
            response.raise_for_status()
            data = response.json()

            log_dict["message"] = "Tabapay: Transaction created"
            log_dict["response"] = json_dumps(data)
            logging.info(json_dumps(log_dict))

            return str(data.get("transactionId", "")), TabapayStatusEnum(str(data.get("status", "")))
        except httpx.HTTPError as e:
            error_details = {
                "request": {
                    "referenceID": request.referenceID,
                    "accounts": request.accounts.__dict__,
                    "currency": request.currency,
                    "amount": str(request.amount),
                    "type": request.type,
                    "memo": request.memo,
                    "purposeOfPayment": request.purposeOfPayment,
                    "achOptions": request.achOptions.value if request.achOptions else None,
                    "achEntryType": request.achEntryType.value if request.achEntryType else None,
                },
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None)
                if isinstance(e, httpx.HTTPStatusError)
                else None,
            }
            raise TabapayError("Error creating transaction in Tabapay", error_details) from e
