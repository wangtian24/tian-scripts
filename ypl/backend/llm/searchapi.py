import os
from typing import cast

import aiohttp

_SEARCHAPI_API_URL = "https://www.searchapi.io/api/v1/search"


async def fetch_search_api_response(params: dict) -> dict:
    api_params = {"api_key": os.getenv("SEARCHAPI_API_KEY")} | params

    async with aiohttp.ClientSession() as session:
        async with session.get(_SEARCHAPI_API_URL, params=api_params) as response:
            return cast(dict, await response.json())
