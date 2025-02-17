from fastapi import APIRouter
from ypl.webhooks.routes.v1 import coinbase, paypal

api_router = APIRouter()

# Include all webhook routes
api_router.include_router(coinbase.router, prefix="/coinbase", tags=["coinbase"])
api_router.include_router(paypal.router, prefix="/paypal", tags=["paypal"])
