from typing import Any


class VendorRegistrationError(Exception):
    pass


class VendorRegistrationResponse:
    def __init__(self, vendor_id: str, additional_details: dict[str, Any], vendor_url_link: str | None = None):
        self.vendor_id = vendor_id
        self.additional_details = additional_details
        self.vendor_url_link = vendor_url_link
