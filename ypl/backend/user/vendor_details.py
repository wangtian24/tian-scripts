from dataclasses import dataclass


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
class AdditionalDetails:
    """Container for various vendor-specific registration details."""

    hyperwallet_details: HyperwalletDetails | None = None
