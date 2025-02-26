from fastapi import APIRouter
from ypl.partner_payments.server.common.types import (
    GetBalanceRequest,
    GetBalanceResponse,
)
from ypl.partner_payments.server.partner.clients import partner_clients
from ypl.partner_payments.server.partner.tabapay.client import (
    TabapayAccountCreationRequest,
    TabapayAccountDetails,
    TabapayCreateAccountResponse,
    TabapayStatusEnum,
    TabapayTransactionRequest,
    TabapayTransactionResponse,
)

JSON_SEPARATORS = (",", ":")  # For compact JSON
router = APIRouter(tags=["tabapay"])


@router.get("/balance")
async def get_balance(request: GetBalanceRequest) -> GetBalanceResponse:
    return await partner_clients.tabapay.get_balance(request)


@router.get("/accounts/{account_id}")
async def get_account_details(account_id: str) -> TabapayAccountDetails:
    return await partner_clients.tabapay.get_account_details(account_id)


@router.get("/transactions/{transaction_id}")
async def get_transaction_status(transaction_id: str) -> TabapayStatusEnum:
    return await partner_clients.tabapay.get_transaction_status(transaction_id)


@router.post("/rtp-details")
async def get_rtp_details(routing_number: str) -> bool:
    return await partner_clients.tabapay.get_rtp_details(routing_number)


@router.post("/accounts")
async def create_account(request: TabapayAccountCreationRequest) -> TabapayCreateAccountResponse:
    return await partner_clients.tabapay.create_account(request)


@router.post("/transactions")
async def create_transaction(request: TabapayTransactionRequest) -> TabapayTransactionResponse:
    return await partner_clients.tabapay.create_transaction(request)
