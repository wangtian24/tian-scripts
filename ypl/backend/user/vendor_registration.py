import abc
import logging
from typing import Any

import httpx
from ypl.backend.config import settings
from ypl.backend.payment.stripe.stripe_payout import StripeRecipientCreateRequest, create_recipient_account
from ypl.backend.user.vendor_details import AdditionalDetails, HyperwalletDetails, StripeDetails
from ypl.backend.utils.json import json_dumps


class VendorRegistrationResponse:
    def __init__(self, vendor_id: str, additional_details: dict[str, Any] | AdditionalDetails):
        self.vendor_id = vendor_id
        self.additional_details = additional_details


class VendorRegistrationError(Exception):
    pass


class VendorRegistration(abc.ABC):
    @abc.abstractmethod
    async def register_user(
        self,
        user_id: str,
        additional_details: AdditionalDetails,
        client: httpx.AsyncClient | None = None,
    ) -> VendorRegistrationResponse:
        """Register a user with the vendor's system.

        Args:
            user_id: The ID of the user to register
            additional_details: Vendor-specific registration details
            client: Optional httpx client to use for the request

        Returns:
            VendorRegistrationResponse containing the vendor's ID for the user and raw response

        Raises:
            VendorRegistrationError: If registration fails
        """
        raise NotImplementedError


class HyperwalletRegistration(VendorRegistration):
    def __init__(self) -> None:
        self.api_url = settings.HYPERWALLET_API_URL
        self.program_token = settings.HYPERWALLET_PROGRAM_TOKEN
        self.username = settings.HYPERWALLET_USERNAME
        self.password = settings.HYPERWALLET_PASSWORD

        if not all([self.program_token, self.username, self.password]):
            raise VendorRegistrationError("Missing required Hyperwallet credentials")

    async def register_user(
        self,
        user_id: str,
        additional_details: AdditionalDetails,
        client: httpx.AsyncClient | None = None,
    ) -> VendorRegistrationResponse:
        """Register a user with Hyperwallet.

        Args:
            user_id: The ID of the user to register
            additional_details: Additional details for Hyperwallet registration containing HyperwalletDetails
            client: Optional httpx client to use for the request

        Returns:
            VendorRegistrationResponse containing the Hyperwallet user token and raw response

        Raises:
            VendorRegistrationError: If Hyperwallet registration fails
        """
        try:
            log_dict: dict[str, Any] = {
                "message": "Hyperwallet: Registering user",
                "user_id": user_id,
                "additional_details": additional_details,
            }
            logging.info(json_dumps(log_dict))

            if isinstance(additional_details.hyperwallet_details, dict):
                additional_details.hyperwallet_details = HyperwalletDetails(**additional_details.hyperwallet_details)

            if not additional_details.hyperwallet_details:
                raise VendorRegistrationError("HyperwalletDetails are required for Hyperwallet registration")

            hw_details = additional_details.hyperwallet_details

            # Prepare the request payload
            payload = {
                "programToken": self.program_token,
                "clientUserId": user_id,
                "profileType": hw_details.profile_type,
                "email": hw_details.email,
                "firstName": hw_details.first_name,
                "lastName": hw_details.last_name,
                "dateOfBirth": hw_details.date_of_birth,
                "addressLine1": hw_details.address_line_1,
                "city": hw_details.city,
                "stateProvince": hw_details.state_province,
                "country": hw_details.country,
                "postalCode": hw_details.postal_code,
            }

            log_dict = {
                "message": "Hyperwallet: User registration payload",
                "payload": payload,
                "api_url": self.api_url,
                "program_token": self.program_token,
                "username": self.username,
                "password": self.password,
            }
            logging.info(json_dumps(log_dict))

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

            # We've already checked these are not None in __init__
            assert self.username is not None and self.password is not None
            auth = (self.username, self.password)

            should_close_client = False
            if client is None:
                client = httpx.AsyncClient()
                should_close_client = True

            try:
                response = await client.post(
                    f"{self.api_url}/users",
                    json=payload,
                    auth=auth,
                    headers=headers,
                )
                response.raise_for_status()
                response_data = response.json()
                token = response_data["token"]
                log_dict = {
                    "message": "Hyperwallet: User registered successfully",
                    "user_id": user_id,
                    "vendor_id": token,
                }
                logging.info(json_dumps(log_dict))

                return VendorRegistrationResponse(
                    vendor_id=token,
                    additional_details=response_data,
                )

            finally:
                if should_close_client:
                    await client.aclose()

        except httpx.HTTPStatusError as e:
            response_data = e.response.json()
            log_dict = {
                "message": "Hyperwallet: API error",
                "status": e.response.status_code,
                "response": response_data,
            }
            logging.error(json_dumps(log_dict))
            raise VendorRegistrationError(f"API error: {response_data.get('message', 'Unknown error')}") from e

        except Exception as e:
            log_dict = {
                "message": "Hyperwallet: General error registering user",
                "user_id": user_id,
                "error": str(e),
            }
            logging.error(json_dumps(log_dict))
            raise VendorRegistrationError(f"Failed to register user with Hyperwallet: {str(e)}") from e


class StripeRegistration(VendorRegistration):
    def __init__(self) -> None:
        pass

    async def register_user(
        self,
        user_id: str,
        additional_details: AdditionalDetails,
        client: httpx.AsyncClient | None = None,
    ) -> VendorRegistrationResponse:
        """Register a user with Stripe.

        Args:
            user_id: The ID of the user to register
            additional_details: Additional details for Stripe registration
            client: Optional httpx client to use for the request (not used for Stripe)

        Returns:
            VendorRegistrationResponse containing the Stripe account ID and raw response

        Raises:
            VendorRegistrationError: If Stripe registration fails
        """
        try:
            log_dict: dict[str, Any] = {
                "message": "Stripe: Registering user",
                "user_id": user_id,
                "additional_details": additional_details,
            }
            logging.info(json_dumps(log_dict))

            if isinstance(additional_details.stripe_details, dict):
                additional_details.stripe_details = StripeDetails(**additional_details.stripe_details)

            if not additional_details.stripe_details:
                raise VendorRegistrationError("StripeDetails are required for Stripe registration")

            stripe_details = additional_details.stripe_details

            request = StripeRecipientCreateRequest(
                given_name=stripe_details.given_name,
                surname=stripe_details.surname,
                email=stripe_details.email,
                country=stripe_details.country,
            )

            account_id = await create_recipient_account(request)

            log_dict = {
                "message": "Stripe: User registered successfully",
                "user_id": user_id,
                "vendor_id": account_id,
            }
            logging.info(json_dumps(log_dict))

            # Convert AdditionalDetails to a dictionary for JSON serialization
            additional_details_dict = {
                "stripe_details": {
                    "given_name": stripe_details.given_name,
                    "surname": stripe_details.surname,
                    "email": stripe_details.email,
                    "country": stripe_details.country,
                }
            }

            return VendorRegistrationResponse(
                vendor_id=account_id,
                additional_details=additional_details_dict,
            )

        except Exception as e:
            log_dict = {
                "message": "Stripe: General error registering user",
                "user_id": user_id,
                "error": str(e),
            }
            logging.error(json_dumps(log_dict))
            raise VendorRegistrationError(f"Failed to register user with Stripe: {str(e)}") from e


def get_vendor_registration(vendor_name: str) -> VendorRegistration:
    """Factory function to get the appropriate vendor registration handler.

    Args:
        vendor_name: Name of the vendor (e.g. "hyperwallet")

    Returns:
        VendorRegistration implementation for the specified vendor

    Raises:
        ValueError: If vendor is not supported
    """
    vendor_map = {
        "hyperwallet": HyperwalletRegistration,
        "stripe": StripeRegistration,
    }

    vendor_class = vendor_map.get(vendor_name.lower())
    if not vendor_class:
        raise ValueError(f"Unsupported vendor: {vendor_name}")

    return vendor_class()
