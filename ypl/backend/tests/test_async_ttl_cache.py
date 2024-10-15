import asyncio
import random
import time
from typing import Any

import pytest

from ypl.utils import async_timed_cache


@async_timed_cache(seconds=1, maxsize=2)
async def cache(**long_args: Any) -> float:
    return time.time()


@pytest.mark.asyncio
async def test_cache_hit_and_miss() -> None:
    long_arg1 = random.sample("abcdefghijklmnopqrstuvwxyz", 26)
    long_arg2 = random.sample("abcdefghijklmnopqrstuvwxyz", 26)

    a = await cache(**{k: 0 for k in long_arg2})
    b = await cache(**{k: 0 for k in long_arg1})

    assert a == b

    await asyncio.sleep(2)

    c = await cache(**{k: 0 for k in long_arg1})
    assert c != a


@pytest.mark.asyncio
async def test_cache_maxsize() -> None:
    await cache(a=1)
    await cache(a=2)
    await cache(a=3)
    await cache(a=4)
    purged = await cache(a=1)
    a = await cache(a=3)
    b = await cache(a=4)
    a2 = await cache(a=3)
    b2 = await cache(a=4)
    purged2 = await cache(a=1)

    assert a == a2 and b == b2 and purged != purged2
