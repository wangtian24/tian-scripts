import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import Column, desc
from sqlmodel import select
from typing_extensions import TypedDict
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


@dataclass
class IPInfoWithModifiedAt:
    ip_info: IPInfo
    modified_at: datetime | None


@dataclass
class UserIPDetailsResponse:
    ip_details: list[IPInfoWithModifiedAt]
    has_more_results: bool


async def get_ip_details(ip_address: str) -> IPInfo | None:
    """Get detailed information about an IP address."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://ipinfo.io/{ip_address}",
                params={"token": settings.IPINFO_API_KEY},
            )
            response.raise_for_status()
            api_response = response.json()

            return {
                "ip": api_response["ip"],
                "hostname": api_response.get("hostname"),
                "city": api_response.get("city"),
                "region": api_response.get("region"),
                "country": api_response.get("country"),
                "loc": api_response.get("loc"),
                "org": api_response.get("org"),
                "postal": api_response.get("postal"),
                "timezone": api_response.get("timezone"),
            }
    except Exception as e:
        log_dict = {
            "message": "Failed to get IP details",
            "error": str(e),
            "ip_address": ip_address,
        }
        logging.warning(json_dumps(log_dict))
        return None


async def store_ip_details(ip_address: str, user_id: str | None = None) -> IPInfo | None:
    """First checks if we have the IP in our database. If not, fetches from ipinfo.io
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
                await session.execute(select(IPs).where((IPs.ip == ip_address) and (IPs.deleted_at is None)))
            ).scalar_one_or_none()

            if ip_details:
                if ip_details.created_at < datetime.now(UTC) - timedelta(days=60):
                    try:
                        ip_info = await get_ip_details(ip_address)
                        if ip_info:
                            ip_details.hostname = ip_info.get("hostname")
                            ip_details.city = ip_info.get("city")
                            ip_details.region = ip_info.get("region")
                            ip_details.country = ip_info.get("country")
                            ip_details.loc = ip_info.get("loc")
                            ip_details.org = ip_info.get("org")
                            ip_details.postal = ip_info.get("postal")
                            ip_details.timezone = ip_info.get("timezone")
                            await session.commit()
                            result_info = ip_info
                        else:
                            result_info = ip_details
                    except Exception as e:
                        log_dict = {
                            "message": "Failed to get IP details",
                            "error": str(e),
                            "ip_address": ip_address,
                        }
                        logging.warning(json_dumps(log_dict))
                        result_info = ip_details
                else:
                    result_info = ip_details
            elif not ip_details:
                try:
                    ip_info = await get_ip_details(ip_address)
                    if ip_info:
                        ip_details = IPs(
                            ip=ip_info.get("ip"),
                            hostname=ip_info.get("hostname"),
                            city=ip_info.get("city"),
                            region=ip_info.get("region"),
                            country=ip_info.get("country"),
                            loc=ip_info.get("loc"),
                            org=ip_info.get("org"),
                            postal=ip_info.get("postal"),
                            timezone=ip_info.get("timezone"),
                        )
                        session.add(ip_details)
                        await session.commit()
                        result_info = ip_info
                    else:
                        return None
                except Exception as e:
                    log_dict = {
                        "message": "Failed to get IP details",
                        "error": str(e),
                        "ip_address": ip_address,
                    }
                    logging.warning(json_dumps(log_dict))
                    return None
            else:
                return None

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
                else:
                    existing_association.modified_at = datetime.now()
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


async def get_user_ip_details(user_id: str, limit: int = 100, offset: int = 0) -> UserIPDetailsResponse | None:
    """Get the IP details for a user."""
    try:
        async with get_async_session() as session:
            user_ip_details = (
                await session.execute(
                    select(IPs, UserIPDetails.modified_at)
                    .join(UserIPDetails, IPs.ip == UserIPDetails.ip)  # type: ignore
                    .where(Column("user_id") == user_id)
                    .order_by(desc(UserIPDetails.modified_at))  # type: ignore
                    .offset(offset)
                    .limit(limit + 1)
                )
            ).all()

            has_more_results = len(user_ip_details) > limit
            user_ip_details = user_ip_details[:limit]

            ip_details = []
            for user_ip_detail in user_ip_details:
                ip_info: IPInfo = {
                    "ip": user_ip_detail[0].ip,
                    "hostname": user_ip_detail[0].hostname,
                    "city": user_ip_detail[0].city,
                    "region": user_ip_detail[0].region,
                    "country": user_ip_detail[0].country,
                    "loc": user_ip_detail[0].loc,
                    "org": user_ip_detail[0].org,
                    "postal": user_ip_detail[0].postal,
                    "timezone": user_ip_detail[0].timezone,
                }
                ip_details.append(IPInfoWithModifiedAt(ip_info=ip_info, modified_at=user_ip_detail[1]))

            return UserIPDetailsResponse(ip_details=ip_details, has_more_results=has_more_results)
    except Exception as e:
        log_dict = {
            "message": "Failed to get user IP details",
            "error": str(e),
            "user_id": user_id,
        }
        logging.warning(json_dumps(log_dict))
        return None
