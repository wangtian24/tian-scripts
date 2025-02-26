import json as json_stdlib
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
JSON_SEPARATORS = (",", ":")  # For compact JSON

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

    CHECKING = "C"
    SAVINGS = "S"


class TabapayAchOptionsEnum(str, Enum):
    """ACH Options for Tabapay transactions."""

    RTP = "R"
    NEXT_DAY = "N"
    SAME_DAY = "S"


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
        settlement_account_id: Account ID for settlement - This is the account from where funds will go out
    """

    api_url: str
    client_id: str
    bearer_token: str
    settlement_account_id: str

    def validate(self) -> None:
        """Validates that all required fields are present."""
        if not self.api_url:
            raise TabapayError("Tabapay API URL is missing")
        if not self.client_id:
            raise TabapayError("Tabapay client ID is missing")
        if not self.bearer_token:
            raise TabapayError("Tabapay bearer token is missing")
        if not self.settlement_account_id:
            raise TabapayError("Tabapay settlement account ID is missing")


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

    countryCode: str
    number: str


@dataclass(frozen=True)
class TabapayOwnerInfo:
    """Complete owner information including name, phone, and optional address."""

    name: TabapayOwnerName
    phone: TabapayOwnerPhone | None = None
    address: TabapayOwnerAddress | None = None


@dataclass(frozen=True)
class TabapayAccountDetails:
    """Complete account details including owner info and optional bank/card details."""

    owner: TabapayOwnerInfo
    bank: TabapayBankInfo | None = None
    card: TabapayCardInfo | None = None


@dataclass(frozen=True)
class TabapayCardCreationInfo:
    """Card details for account creation. We only use the token for the card."""

    token: str


@dataclass(frozen=True)
class TabapayAccountCreationRequest:
    """Request object for creating a Tabapay account.

    Attributes:
        referenceID: Unique identifier for the account creation
        card: Optional card details
        bank: Optional bank details

    Raises:
        TabapayError: If neither card nor bank details are provided
    """

    referenceID: str
    owner: TabapayOwnerInfo
    card: TabapayCardCreationInfo | None = None
    bank: TabapayBankInfo | None = None

    def __post_init__(self) -> None:
        """Validates that at least one of card or bank is populated."""
        if self.card is None and self.bank is None:
            raise TabapayError("Either card or bank details must be provided")


@dataclass(frozen=True)
class TabapayTransactionFees:
    """Fee structure for a Tabapay transaction."""

    interchange: Decimal
    network: Decimal
    tabapay: Decimal


@dataclass(frozen=True)
class TabapayTransactionResponse:
    """Response object for a Tabapay transaction.

    Attributes:
        transactionID: Unique identifier for the transaction
        network: Network used for the transaction (e.g. "Visa")
        networkRC: Network response code
        networkID: Network transaction identifier
        status: Status of the transaction
        approvalCode: Transaction approval code
        fees: Transaction fee details
        additional: Additional transaction metadata (e.g. PAR)
        card: Card details if transaction involves a card
    """

    transactionID: str
    network: str
    networkRC: str
    networkID: str
    status: TabapayStatusEnum
    approvalCode: str
    fees: TabapayTransactionFees | None = None
    additional: dict[str, str] | None = None
    card: TabapayCardInfo | None = None


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
    amount: Decimal
    currency: str
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
        logging.error(json_dumps(log_dict, separators=JSON_SEPARATORS))


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
            api_url=self.config["api_url"],
            client_id=self.config["client_id"],
            bearer_token=self.config["bearer_token"],
            settlement_account_id=self.config["settlement_account_id"],
        )
        config_obj.validate()

        self._config_obj = config_obj
        self.http_client = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT_SECONDS,
            limits=httpx.Limits(max_keepalive_connections=MAX_RETRIES),
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
        return {
            "accept": CONTENT_TYPE_JSON,
            "content-type": CONTENT_TYPE_JSON,
            "authorization": f"{AUTH_BEARER} {self._config_obj.bearer_token}",
        }

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
        logging.info(json_dumps({"message": "Balance info not available for Tabapay"}, separators=JSON_SEPARATORS))
        return GetBalanceResponse(balance=Decimal(0), ip_address="unknown")

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
        logging.info(
            json_dumps(
                {"message": "fetching rtp details from tabapay", "routing_number": routing_number},
                separators=JSON_SEPARATORS,
            )
        )
        url = self._get_base_url("banks")
        assert self.http_client is not None

        try:
            response = await self.http_client.post(
                url, headers=self._get_headers(), json={"routingNumber": routing_number}
            )
            response.raise_for_status()
            data = response.json()
            logging.info(
                json_dumps(
                    {
                        "message": "fetched rtp details from tabapay",
                        "response": json_dumps(data, separators=JSON_SEPARATORS),
                    },
                    separators=JSON_SEPARATORS,
                )
            )
            return bool(data.get("RTP", False))
        except httpx.HTTPError as e:
            error_details = {
                "routing_number": routing_number,
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None)
                if isinstance(e, httpx.HTTPStatusError)
                else None,
            }
            logging.warning(json_dumps(error_details, separators=JSON_SEPARATORS))
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
        logging.info(
            json_dumps(
                {"message": "fetching account details from tabapay", "account_id": account_id},
                separators=JSON_SEPARATORS,
            )
        )

        url = self._get_base_url(f"accounts/{account_id}")
        assert self.http_client is not None

        try:
            response = await self.http_client.get(url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()

            logging.info(
                json_dumps(
                    {"message": "fetched account details from tabapay", "status_code": response.status_code},
                    separators=JSON_SEPARATORS,
                )
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
                phone=TabapayOwnerPhone(
                    number=owner_data.get("phone", ""), countryCode=owner_data.get("countryCode", "")
                ),
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
        logging.info(
            json_dumps(
                {"message": "Tabapay: fetching transaction status", "transaction_id": transaction_id},
                separators=JSON_SEPARATORS,
            )
        )

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
                        "data": json_dumps(data, separators=JSON_SEPARATORS),
                    },
                    separators=JSON_SEPARATORS,
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
    async def create_account(self, request: TabapayAccountCreationRequest) -> TabapayCreateAccountResponse:
        """Create an account in Tabapay.

        Args:
            request: Account details for creation

        Returns:
            TabapayCreateAccountResponse containing the account ID and additional metadata

        Raises:
            TabapayError: If account creation fails
        """
        request_dict: dict[str, Any] = {
            "referenceID": request.referenceID,
            "owner": {
                "name": {
                    "first": request.owner.name.first,
                    "last": request.owner.name.last,
                },
                "phone": {"number": request.owner.phone.number} if request.owner.phone else None,
            },
        }

        # Add optional address if present
        if request.owner.address:
            request_dict["owner"]["address"] = {
                "line1": request.owner.address.line1,
                "line2": request.owner.address.line2,
                "city": request.owner.address.city,
                "state": request.owner.address.state,
                "zipcode": request.owner.address.zipcode,
            }

        # Add bank info if present
        if request.bank:
            request_dict["bank"] = {
                "routingNumber": request.bank.routingNumber,
                "accountNumber": request.bank.accountNumber,
                "accountType": request.bank.accountType.value,
            }

        # Add card info if present
        if request.card:
            request_dict["card"] = {
                "token": request.card.token,
            }

        log_dict = {
            "message": "Tabapay: Creating account",
            "data": request_dict,
        }
        logging.info(json_dumps(log_dict, separators=JSON_SEPARATORS))

        url = self._get_base_url("accounts")
        assert self.http_client is not None

        try:
            response = await self.http_client.post(
                url,
                headers=self._get_headers(),
                content=json_stdlib.dumps(request_dict, separators=JSON_SEPARATORS).encode(),
            )
            response.raise_for_status()
            data = response.json()

            log_dict["message"] = "Tabapay: Account created"
            log_dict["data"] = json_dumps(data, separators=JSON_SEPARATORS)
            logging.info(json_dumps(log_dict, separators=JSON_SEPARATORS))

            # Extract metadata (everything except standard fields)
            metadata = {k: v for k, v in data.items() if k not in ["SC", "EC", "accountID"]}

            return TabapayCreateAccountResponse(
                account_id=str(data.get("accountID", "")),
                metadata=metadata,
            )
        except httpx.HTTPError as e:
            error_details = {
                "request": request_dict,
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None)
                if isinstance(e, httpx.HTTPStatusError)
                else None,
            }
            raise TabapayError("Error creating account in Tabapay", error_details) from e

    @require_initialization
    async def create_transaction(self, request: TabapayTransactionRequest) -> TabapayTransactionResponse:
        """Create a transaction in Tabapay.

        Args:
            request: Transaction details for creation

        Returns:
            TabapayTransactionResponse containing the transaction details and status

        Raises:
            TabapayError: If transaction creation fails
        """
        if self._config_obj is None or self._config_obj.settlement_account_id is None:
            raise TabapayError("Tabapay client not initialized")

        request_dict = {
            "referenceID": request.referenceID,
            "accounts": {
                "sourceAccountID": self._config_obj.settlement_account_id,
                "destinationAccountID": request.accounts.destinationAccountID,
            },
            "currency": request.currency,
            "amount": str(request.amount),
            "type": request.type,
        }

        # Add optional fields if present
        if request.memo is not None:
            request_dict["memo"] = request.memo
        if request.purposeOfPayment is not None:
            request_dict["purposeOfPayment"] = request.purposeOfPayment
        if request.achOptions is not None:
            request_dict["achOptions"] = request.achOptions.value
        if request.achEntryType is not None:
            request_dict["achEntryType"] = request.achEntryType.value

        log_dict = {
            "message": "Tabapay: Creating transaction",
            "request": request_dict,
        }
        logging.info(json_dumps(log_dict, separators=JSON_SEPARATORS))

        url = self._get_base_url("transactions")
        assert self.http_client is not None

        try:
            response = await self.http_client.post(
                url,
                headers=self._get_headers(),
                content=json_stdlib.dumps(request_dict, separators=JSON_SEPARATORS).encode(),
            )
            response.raise_for_status()
            data = response.json()

            log_dict["message"] = "Tabapay: Transaction created"
            log_dict["response"] = json_dumps(data, separators=JSON_SEPARATORS)
            logging.info(json_dumps(log_dict, separators=JSON_SEPARATORS))

            # Parse card info if present
            card_info = None
            if card_data := data.get("card"):
                card_info = TabapayCardInfo(
                    last4=card_data.get("last4", ""),
                    expirationDate=card_data.get("expirationDate", ""),
                )

            return TabapayTransactionResponse(
                transactionID=str(data.get("transactionID", "")),
                network=str(data.get("network", "")),
                networkRC=str(data.get("networkRC", "")),
                networkID=str(data.get("networkID", "")),
                status=TabapayStatusEnum(str(data.get("status", ""))),
                approvalCode=str(data.get("approvalCode", "")),
                additional=data.get("additional"),
                card=card_info,
            )
        except httpx.HTTPError as e:
            error_details = {
                "request": request_dict,
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None)
                if isinstance(e, httpx.HTTPStatusError)
                else None,
            }
            raise TabapayError("Error creating transaction in Tabapay", error_details) from e
