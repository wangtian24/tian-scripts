from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class HyperwalletDetails:
    """Details required for Hyperwallet user registration."""

    email: str
    first_name: str
    last_name: str
    date_of_birth: str  # Format: YYYY-MM-DD
    address_line_1: str
    city: str
    state_province: str
    country: str  # Country code (e.g. US)
    postal_code: str
    profile_type: str = "INDIVIDUAL"


@dataclass
class StripeDetails:
    """Details required for Stripe user registration."""

    given_name: str
    surname: str
    email: str
    country: str  # Country code (e.g. US)


@dataclass
class AdditionalDetails:
    """Container for various vendor-specific registration details."""

    hyperwallet_details: HyperwalletDetails | None = None
    stripe_details: StripeDetails | None = None


class StripeDetailsRequest(BaseModel):
    """Request model for Stripe registration details."""

    given_name: str = Field(..., description="Given name of the user")
    surname: str = Field(..., description="Surname of the user")
    email: str = Field(..., description="Email address of the user")
    country: str = Field(..., description="Country code (e.g. US)")


class AdditionalDetailsRequest(BaseModel):
    """Request model for vendor-specific registration details."""

    stripe_details: StripeDetailsRequest | None = Field(None, description="Details for Stripe registration")
    hyperwallet_details: dict[str, Any] | None = Field(None, description="Details for Hyperwallet registration")


class VendorRegistrationRequest(BaseModel):
    """Request model for vendor registration."""

    user_id: str = Field(..., description="ID of the user to register")
    vendor_name: str = Field(..., description="Name of the vendor (e.g. stripe, hyperwallet)")
    additional_details: AdditionalDetailsRequest = Field(..., description="Vendor-specific registration details")
