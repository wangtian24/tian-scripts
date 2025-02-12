from fastapi import APIRouter, Depends
from ypl.partner_payments.server.routes.api_auth import validate_api_key
from ypl.partner_payments.server.routes.v1 import axis

api_router = APIRouter(dependencies=[Depends(validate_api_key)])

# Include all webhook routes
api_router.include_router(axis.router, prefix="/axis", tags=["axis"])
