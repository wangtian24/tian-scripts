import logging
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

import httpx
from pydantic import BaseModel, Field
from ypl.backend.config import settings
from ypl.backend.utils.json import json_dumps
from ypl.partner_payments.server.common.types import GetBalanceRequest, GetBalanceResponse
from ypl.partner_payments.server.partner.tabapay.client import (
    TabapayAccountDetails,
    TabapayCreateAccountResponse,
    TabapayTransactionRequest,
)


@dataclass
class TabaPayProxyConfig:
    """Configuration for TabaPay integration."""

    api_url: str = Field(settings.partner_payments_api_url)
    api_key: str = Field(settings.X_API_KEY)
    timeout: int = 30
    memo_template: str = "Payout from YUPP! Thanks for using YUPP!"


class TabaPayResponse(BaseModel):
    """Base model for TabaPay API responses."""

    transaction_id: str | None = None
    status: str | None = None
    error: str | None = None


class TabaPayPayoutError(Exception):
    """Custom exception for TabaPay payout related errors."""

    def __init__(
        self,
        message: str = "Internal error",
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.details = details or {}
        self.status_code = status_code
        self._log_error()

    def _log_error(self) -> None:
        """Log the error with details."""
        log_dict: dict[str, Any] = {"message": str(self), "details": self.details, "status_code": self.status_code}
        logging.error(json_dumps(log_dict))


class TabaPayClient:
    """Client for handling TabaPay API interactions."""

    GENERIC_ERROR_MESSAGE: Final[str] = "Internal error"

    def __init__(self, config: TabaPayProxyConfig | None = None):
        """Initialize TabaPay client with configuration."""
        self.config = config or TabaPayProxyConfig()
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Initialize the HTTP client if not already initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout, limits=httpx.Limits(max_keepalive_connections=5)
            )

    async def cleanup(self) -> None:
        """Clean up the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "TabaPayClient":
        """Set up async context manager."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any | None) -> None:
        """Clean up async context manager."""
        await self.cleanup()

    def _get_headers(self) -> dict[str, str]:
        """Get common headers for API requests."""
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.config.api_key,
        }

    def _validate_client(self) -> None:
        """Validate that the client is properly initialized."""
        if not self._client:
            raise TabaPayPayoutError("Client not initialized. Use async context manager.")

    def _get_client(self) -> httpx.AsyncClient:
        """Get the HTTP client, raising an error if not initialized."""
        self._validate_client()
        assert self._client is not None
        return self._client

    async def process_payout(
        self,
        tabapay_request: TabapayTransactionRequest,
    ) -> tuple[str, str]:
        """Process a payout transaction."""
        client = self._get_client()

        log_dict: dict[str, Any] = {
            "message": "TabaPay: Processing payout",
            "tabapay_request": asdict(tabapay_request),
        }

        logging.info(json_dumps(log_dict))

        if not all(
            [
                tabapay_request.amount,
                tabapay_request.accounts.sourceAccountID,
                tabapay_request.currency,
                tabapay_request.referenceID,
                tabapay_request.accounts.destinationAccountID,
            ]
        ):
            validation_details: dict[str, Any] = {
                "message": "TabaPay: Missing required fields",
                "tabapay_request": asdict(tabapay_request),
                "error": "Missing required fields",
            }
            raise TabaPayPayoutError(self.GENERIC_ERROR_MESSAGE, validation_details)

        # Round the amount to 2 decimal places for fiat currency
        rounded_amount = tabapay_request.amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Create new request with rounded amount and memo
        request_dict = asdict(tabapay_request)
        request_dict.update(
            {
                "amount": rounded_amount,
                "memo": request_dict.get("memo") or f"{self.config.memo_template} - {tabapay_request.referenceID}",
            }
        )

        try:
            response = await client.post(
                f"{self.config.api_url}/v1/tabapay/transactions",
                json=request_dict,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()

            transaction_id = data.get("transaction_id")
            status = data.get("status")

            if not transaction_id:
                details = {
                    "payment_transaction_id": str(transaction_id),
                    "error": "TabaPay: Missing transaction_id in payment response",
                }
                raise TabaPayPayoutError(self.GENERIC_ERROR_MESSAGE, details)

            log_dict = {
                "message": "TabaPay: Payout created",
                "amount": str(tabapay_request.amount),
                "currency": str(tabapay_request.currency),
                "payment_transaction_id": str(tabapay_request.referenceID),
                "destination_type": str(tabapay_request.accounts.destinationAccountID),
                "destination_identifier": str(tabapay_request.accounts.destinationAccountID),
                "transaction_id": transaction_id,
                "status": status,
            }
            logging.info(json_dumps(log_dict))

            return transaction_id, status

        except httpx.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, "response") else None
            details = {
                "message": "TabaPay: Error processing payout",
                "payment_transaction_id": str(tabapay_request.referenceID),
                "error": str(e),
            }
            raise TabaPayPayoutError(str(e), details, status_code) from e
        except Exception as e:
            details = {
                "message": "TabaPay: Error processing payout",
                "payment_transaction_id": str(tabapay_request.referenceID),
                "error": str(e),
            }
            raise TabaPayPayoutError(str(e), details) from e

    async def get_transaction_status(self, transaction_id: str) -> str:
        """Get the status of a specific transaction."""
        client = self._get_client()

        try:
            response = await client.get(
                f"{self.config.api_url}/v1/tabapay/transactions/{transaction_id}/status",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()

            status: str = str(data.get("status", "UNKNOWN"))

            log_dict = {
                "message": "TabaPay: Retrieved transaction status",
                "transaction_id": transaction_id,
                "status": status,
            }
            logging.info(json_dumps(log_dict))

            return status

        except httpx.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, "response") else None
            details = {
                "transaction_id": transaction_id,
                "error": str(e),
            }
            raise TabaPayPayoutError("TabaPay: Error getting transaction status", details, status_code) from e
        except Exception as e:
            details = {
                "transaction_id": transaction_id,
                "error": str(e),
            }
            raise TabaPayPayoutError("TabaPay: Error getting transaction status", details) from e

    async def get_balance(self, request: GetBalanceRequest) -> GetBalanceResponse:
        """Get the balance for an account."""
        client = self._get_client()

        try:
            response = await client.post(
                f"{self.config.api_url}/v1/tabapay/balance",
                json=asdict(request),
                headers=self._get_headers(),
            )
            response.raise_for_status()
            return GetBalanceResponse(**response.json())

        except httpx.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, "response") else None
            details = {"error": str(e)}
            raise TabaPayPayoutError("TabaPay: Error getting balance", details, status_code) from e
        except Exception as e:
            details = {"error": str(e)}
            raise TabaPayPayoutError("TabaPay: Error getting balance", details) from e

    async def get_account_details(self, account_id: str) -> TabapayAccountDetails:
        """Get details for a specific account."""
        client = self._get_client()

        try:
            response = await client.get(
                f"{self.config.api_url}/v1/tabapay/account-details/{account_id}",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            return TabapayAccountDetails(**response.json())

        except httpx.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, "response") else None
            details = {
                "account_id": account_id,
                "error": str(e),
            }
            raise TabaPayPayoutError("TabaPay: Error getting account details", details, status_code) from e
        except Exception as e:
            details = {
                "account_id": account_id,
                "error": str(e),
            }
            raise TabaPayPayoutError("TabaPay: Error getting account details", details) from e

    async def create_account(self, account_details: TabapayAccountDetails) -> TabapayCreateAccountResponse:
        """Create a new account."""
        client = self._get_client()

        try:
            response = await client.post(
                f"{self.config.api_url}/v1/tabapay/create-account",
                json=asdict(account_details),
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()

            return TabapayCreateAccountResponse(**data)

        except httpx.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, "response") else None
            details = {"error": str(e)}
            raise TabaPayPayoutError("TabaPay: Error creating account", details, status_code) from e
        except Exception as e:
            details = {"error": str(e)}
            raise TabaPayPayoutError("TabaPay: Error creating account", details) from e

    async def get_rtp_details(self, routing_number: str) -> bool:
        """Get the RTP details for a specific routing number."""
        client = self._get_client()

        try:
            response = await client.post(
                f"{self.config.api_url}/v1/tabapay/rtp-details",
                json={"routingNumber": routing_number},
                headers=self._get_headers(),
            )
            response.raise_for_status()
            return bool(response.json())

        except httpx.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, "response") else None
            details = {"error": str(e)}
            raise TabaPayPayoutError("TabaPay: Error getting RTP details", details, status_code) from e
        except Exception as e:
            details = {"error": str(e)}
            raise TabaPayPayoutError("TabaPay: Error getting RTP details", details) from e
