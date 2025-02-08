"""Utility functions for interacting with the Hyperwallet API."""

import logging
from typing import cast

import httpx
from fastapi import HTTPException
from sqlalchemy import select
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.users import UserVendorProfile, VendorNameEnum


class HyperwalletUtilsError(Exception):
    """Base exception for Hyperwallet utils errors."""

    pass


async def get_hyperwallet_user_auth_token(user_id: str) -> str:
    """Get an authentication token for a Hyperwallet user.

    Args:
        user_id: The ID of the user to get a token for

    Returns:
        str: The authentication token

    Raises:
        HyperwalletUtilsError: If the user is not found or not registered with Hyperwallet
        HTTPException: If the Hyperwallet API request fails
    """
    async with get_async_session() as session:
        query = select(UserVendorProfile).where(
            UserVendorProfile.user_id == user_id,  # type: ignore
            UserVendorProfile.vendor_name == VendorNameEnum.HYPERWALLET,  # type: ignore
            UserVendorProfile.deleted_at.is_(None),  # type: ignore
        )
        result = await session.execute(query)
        user_vendor = result.scalar_one_or_none()

        if not user_vendor:
            log_dict = {
                "message": "Hyperwallet: User token not found",
                "user_id": user_id,
            }
            logging.error(json_dumps(log_dict))
            raise HyperwalletUtilsError("Hyperwallet: User not registered with Hyperwallet")

    # Get Hyperwallet API credentials
    api_username = settings.hyperwallet_username
    api_password = settings.hyperwallet_password
    api_url = settings.hyperwallet_api_url

    if not all([api_username, api_password, api_url]):
        log_dict = {
            "message": "Hyperwallet: Missing API credentials",
            "user_id": user_id,
        }
        logging.warning(json_dumps(log_dict))
        raise HyperwalletUtilsError("Hyperwallet: Missing API credentials")

    # Make request to get authentication token
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    auth = (api_username, api_password)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{api_url}/users/{user_vendor.user_vendor_id}/authentication-token", headers=headers, auth=auth
            )
            response.raise_for_status()

            data = response.json()
            token = data.get("value")

            if not token or not isinstance(token, str):
                log_dict = {
                    "message": "Hyperwallet: No token in response",
                    "user_id": user_id,
                    "response": str(data),
                }
                logging.warning(json_dumps(log_dict))
                raise HyperwalletUtilsError("Hyperwallet: No authentication token in response")

            log_dict = {
                "message": "Hyperwallet: Got authentication token",
                "user_id": user_id,
            }
            logging.info(json_dumps(log_dict))

            return cast(str, token)

    except Exception as e:
        log_dict = {
            "message": "Hyperwallet: Unexpected error getting authentication token",
            "user_id": user_id,
            "error": str(e),
        }
        logging.warning(json_dumps(log_dict))
        raise HTTPException(
            status_code=500, detail=f"Hyperwallet: Unexpected error getting authentication token: {str(e)}"
        ) from e
