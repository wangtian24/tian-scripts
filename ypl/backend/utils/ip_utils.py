import logging
from datetime import datetime, timedelta
from typing import TypedDict

import httpx
from sqlalchemy import Column
from sqlmodel import select
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.users import IPs, UserIPDetails


class IPInfo(TypedDict):
    """Information about an IP address from ipinfo.io."""

    ip: str
    hostname: str | None
    city: str | None
    region: str | None
    country: str | None
    loc: str | None
    org: str | None
    postal: str | None
    timezone: str | None


async def store_ip_details(ip_address: str, user_id: str | None = None) -> IPInfo | None:
    """Get detailed information about an IP address.

    First checks if we have the IP in our database. If not, fetches from ipinfo.io
    and stores in our database. If user_id is provided, also creates an association
    between the user and IP.

    Args:
        ip_address: The IP address to check
        user_id: Optional user ID to associate with this IP

    Returns:
        IPInfo object containing location and network information,
        or None if the API call fails and IP is not in database
    """
    try:
        result_info: IPInfo | None = None
        async with get_async_session() as session:
            ip_details = (
                await session.execute(select(IPs).where((Column("ip") == ip_address) & (Column("deleted_at") is None)))
            ).scalar_one_or_none()

            if ip_details and ip_details.created_at < datetime.now() - timedelta(days=60):
                IPINFO_API_KEY = settings.ipinfo_api_key
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"https://ipinfo.io/{ip_address}",
                            params={"token": IPINFO_API_KEY},
                        )
                        response.raise_for_status()
                        api_response = response.json()

                        # Update existing record
                        ip_details.hostname = api_response.get("hostname")
                        ip_details.city = api_response.get("city")
                        ip_details.region = api_response.get("region")
                        ip_details.country = api_response.get("country")
                        ip_details.loc = api_response.get("loc")
                        ip_details.org = api_response.get("org")
                        ip_details.postal = api_response.get("postal")
                        ip_details.timezone = api_response.get("timezone")
                        await session.commit()

                        result_info = {
                            "ip": ip_details.ip,
                            "hostname": ip_details.hostname,
                            "city": ip_details.city,
                            "region": ip_details.region,
                            "country": ip_details.country,
                            "loc": ip_details.loc,
                            "org": ip_details.org,
                            "postal": ip_details.postal,
                            "timezone": ip_details.timezone,
                        }

                except Exception as e:
                    log_dict = {
                        "message": "Failed to get IP details",
                        "error": str(e),
                        "ip_address": ip_address,
                    }
                    logging.warning(json_dumps(log_dict))
                    return None
            elif not ip_details:
                IPINFO_API_KEY = settings.ipinfo_api_key
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"https://ipinfo.io/{ip_address}",
                            params={"token": IPINFO_API_KEY},
                        )
                        response.raise_for_status()
                        api_response = response.json()

                        db_ip = IPs(
                            ip=api_response["ip"],
                            hostname=api_response.get("hostname"),
                            city=api_response.get("city"),
                            region=api_response.get("region"),
                            country=api_response.get("country"),
                            loc=api_response.get("loc"),
                            org=api_response.get("org"),
                            postal=api_response.get("postal"),
                            timezone=api_response.get("timezone"),
                        )
                        session.add(db_ip)
                        await session.commit()

                        result_info = {
                            "ip": db_ip.ip,
                            "hostname": db_ip.hostname,
                            "city": db_ip.city,
                            "region": db_ip.region,
                            "country": db_ip.country,
                            "loc": db_ip.loc,
                            "org": db_ip.org,
                            "postal": db_ip.postal,
                            "timezone": db_ip.timezone,
                        }

                except Exception as e:
                    log_dict = {
                        "message": "Failed to get IP details",
                        "error": str(e),
                        "ip_address": ip_address,
                    }
                    logging.warning(json_dumps(log_dict))
                    return None
            else:
                # IP exists and is not old, use existing data
                result_info = {
                    "ip": ip_details.ip,
                    "hostname": ip_details.hostname,
                    "city": ip_details.city,
                    "region": ip_details.region,
                    "country": ip_details.country,
                    "loc": ip_details.loc,
                    "org": ip_details.org,
                    "postal": ip_details.postal,
                    "timezone": ip_details.timezone,
                }

            if user_id:
                existing_association = (
                    await session.execute(
                        select(UserIPDetails).where((Column("user_id") == user_id) & (Column("ip") == ip_address))
                    )
                ).scalar_one_or_none()

                if not existing_association:
                    user_ip = UserIPDetails(user_id=user_id, ip=ip_address)
                    session.add(user_ip)
                    await session.commit()

            return result_info

    except Exception as e:
        log_dict = {
            "message": "Failed to store IP details",
            "error": str(e),
            "ip_address": ip_address,
        }
        logging.warning(json_dumps(log_dict))
        return None
