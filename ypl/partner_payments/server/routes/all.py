from fastapi import APIRouter, Depends
from ypl.partner_payments.server.routes.api_auth import validate_api_key
from ypl.partner_payments.server.routes.v1 import axis, tabapay

api_router = APIRouter(dependencies=[Depends(validate_api_key)])

api_router.include_router(axis.router, prefix="/axis", tags=["axis"])
api_router.include_router(tabapay.router, prefix="/tabapay", tags=["tabapay"])
