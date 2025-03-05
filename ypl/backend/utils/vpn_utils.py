import logging
from typing import TypedDict

import httpx
from ypl.backend.config import settings
from ypl.backend.utils.json import json_dumps


class SecurityInfo(TypedDict):
    """Security information about an IP address."""

    vpn: bool
    proxy: bool
    tor: bool
    relay: bool


class LocationInfo(TypedDict):
    """Location information about an IP address."""

    city: str
    region: str
    country: str
    continent: str
    region_code: str
    country_code: str
    continent_code: str
    latitude: str
    longitude: str
    time_zone: str
    locale_code: str
    metro_code: str
    is_in_european_union: bool


class NetworkInfo(TypedDict):
    """Network information about an IP address."""

    network: str
    autonomous_system_number: str
    autonomous_system_organization: str


class IPDetails(TypedDict):
    """Complete details about an IP address."""

    ip: str
    security: SecurityInfo
    location: LocationInfo
    network: NetworkInfo


async def get_ip_details(ip_address: str) -> IPDetails | None:
    """Get detailed information about an IP address using vpnapi.io.

    Args:
        ip_address: The IP address to check

    Returns:
        IPDetails object containing security, location, and network information,
        or None if the API call fails
    """
    VPNAPI_API_KEY = settings.VPNAPI_API_KEY
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://vpnapi.io/api/{ip_address}",
                params={"key": VPNAPI_API_KEY},
            )
            response.raise_for_status()
            data = response.json()

            # Convert the response to our typed structure
            ip_details: IPDetails = {
                "ip": data["ip"],
                "security": {
                    "vpn": bool(data["security"]["vpn"]),
                    "proxy": bool(data["security"]["proxy"]),
                    "tor": bool(data["security"]["tor"]),
                    "relay": bool(data["security"]["relay"]),
                },
                "location": {
                    "city": data["location"]["city"],
                    "region": data["location"]["region"],
                    "country": data["location"]["country"],
                    "continent": data["location"]["continent"],
                    "region_code": data["location"]["region_code"],
                    "country_code": data["location"]["country_code"],
                    "continent_code": data["location"]["continent_code"],
                    "latitude": data["location"]["latitude"],
                    "longitude": data["location"]["longitude"],
                    "time_zone": data["location"]["time_zone"],
                    "locale_code": data["location"]["locale_code"],
                    "metro_code": data["location"]["metro_code"],
                    "is_in_european_union": bool(data["location"]["is_in_european_union"]),
                },
                "network": {
                    "network": data["network"]["network"],
                    "autonomous_system_number": data["network"]["autonomous_system_number"],
                    "autonomous_system_organization": data["network"]["autonomous_system_organization"],
                },
            }

            return ip_details

    except Exception as e:
        log_dict = {
            "message": "Failed to get IP details",
            "error": str(e),
            "ip_address": ip_address,
        }
        logging.error(json_dumps(log_dict))
        return None
