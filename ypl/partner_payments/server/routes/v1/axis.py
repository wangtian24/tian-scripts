from fastapi import APIRouter
from ypl.partner_payments.server.common.types import GetBalanceRequest, GetBalanceResponse
from ypl.partner_payments.server.partner.clients import partner_clients

router = APIRouter(tags=["axis"])


@router.post("/balance")
async def fetch_balance(request: GetBalanceRequest) -> GetBalanceResponse:
    return await partner_clients.axis.get_balance(request)
