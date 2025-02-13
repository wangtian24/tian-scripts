import logging

from ypl.partner_payments.server.partner.axis.client import AxisClient
from ypl.partner_payments.server.partner.tabapay.client import TabaPayClient


class PartnerClients:
    def __init__(self) -> None:
        self.axis = AxisClient()
        self.tabapay = TabaPayClient()
        self.clients = [self.axis, self.tabapay]

    async def initialize(self) -> None:
        logging.info("Initializing partner clients")
        for client in self.clients:
            logging.info(f"Initializing {client.__class__.__name__}")
            await client.initialize()
            logging.info(f"Initialized {client.__class__.__name__}")
        logging.info("Initialized all partner clients")

    async def cleanup(self) -> None:
        logging.info("Cleaning up partner clients")
        for client in self.clients:
            logging.info(f"Cleaning up {client.__class__.__name__}")
            await client.cleanup()
            logging.info(f"Cleaned up {client.__class__.__name__}")
        logging.info("Cleaned up all partner clients")


partner_clients = PartnerClients()
