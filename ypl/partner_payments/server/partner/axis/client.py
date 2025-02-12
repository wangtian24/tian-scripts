import logging
from decimal import Decimal

import httpx
from ypl.backend.utils.json import json_dumps
from ypl.partner_payments.server.common.types import GetBalanceRequest, GetBalanceResponse
from ypl.partner_payments.server.config import secret_manager
from ypl.partner_payments.server.partner.base import BasePartnerClient


class AxisClient(BasePartnerClient):
    def __init__(self) -> None:
        super().__init__()
        self.http_client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        self.config = await secret_manager.get_axis_upi_config()
        self.http_client = httpx.AsyncClient()

    async def cleanup(self) -> None:
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

    async def get_balance(self, request: GetBalanceRequest) -> GetBalanceResponse:
        logging.info(json_dumps({"message": "fetching balance from axis"}))
        if not self.http_client:
            raise Exception("HTTP client not initialized")
        try:
            response = await self.http_client.get("https://curlmyip.org/")
            ip_address = response.text
        except Exception as e:
            logging.error(json_dumps({"message": "error fetching ip address", "error": str(e)}))
            ip_address = "unknown"
        logging.info(json_dumps({"message": "fetched balance from axis"}))
        return GetBalanceResponse(balance=Decimal(1000), ip_address=ip_address)
