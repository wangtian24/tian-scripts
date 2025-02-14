from fastapi import APIRouter
from ypl.partner_payments.server.common.types import (
    GetBalanceRequest,
    GetBalanceResponse,
)
from ypl.partner_payments.server.partner.clients import partner_clients
from ypl.partner_payments.server.partner.tabapay.client import (
    TabapayAccountDetails,
    TabapayCreateAccountResponse,
    TabapayStatusEnum,
    TabapayTransactionRequest,
)

router = APIRouter(tags=["tabapay"])


@router.get("/balance")
async def get_balance(request: GetBalanceRequest) -> GetBalanceResponse:
    return await partner_clients.tabapay.get_balance(request)


@router.get("/account-details/{account_id}")
async def get_account_details(account_id: str) -> TabapayAccountDetails:
    return await partner_clients.tabapay.get_account_details(account_id)


@router.get("/transaction-status/{transaction_id}")
async def get_transaction_status(transaction_id: str) -> TabapayStatusEnum:
    return await partner_clients.tabapay.get_transaction_status(transaction_id)


@router.post("/create-account")
async def create_account(request: TabapayAccountDetails) -> TabapayCreateAccountResponse:
    return await partner_clients.tabapay.create_account(request)


@router.post("/create-transaction")
async def create_transaction(request: TabapayTransactionRequest) -> tuple[str, TabapayStatusEnum]:
    return await partner_clients.tabapay.create_transaction(request)
