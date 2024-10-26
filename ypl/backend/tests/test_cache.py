import asyncio
import random
import threading
import time
from typing import Any

import pytest

from ypl.backend.rw_cache import DummyCache
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


def test_rw_cache() -> None:
    # Test that the read- and write-through cache works as expected.
    cache = DummyCache()
    assert cache.full_read is False
    assert cache.full_write is False

    a = time.time()
    # This takes 1 second in the bkgd and should not block. The key is not specified,
    # so it will be set to the dummy key instead of setting the cache immediately.
    cache.write(value=1)
    assert time.time() - a < 0.5
    assert not cache.full_write

    a = time.time()
    assert cache.read(key=cache.DUMMY_KEY, wait_for_write_secs=0.01) == 5  # this will block for 2 seconds
    assert time.time() - a >= 1.5
    assert cache.full_read
    assert cache.full_write

    a = time.time()
    assert cache.read(key=cache.DUMMY_KEY, wait_for_write_secs=0.01) == 1  # write complete, this should not block
    assert time.time() - a < 0.5


def test_wait_for_write_rw_cache() -> None:
    # Test wait_for_write
    cache = DummyCache()
    cache.write(value=1)
    a = time.time()
    assert cache.read(key=cache.DUMMY_KEY, wait_for_write_secs=5) == 1
    assert time.time() - a <= 4  # 1 second for the write


def test_multithreaded_rw_cache() -> None:
    # Test that the read- and write-through cache doesn't throw an error with multiple threads.
    def thread_run_read() -> None:
        for _ in range(100):
            cache.read(key=cache.DUMMY_KEY, wait_for_write_secs=0.01)
            time.sleep(0.01)

    def thread_run_write() -> None:
        for idx in range(100):
            cache.write(value=idx)
            time.sleep(0.001)

    cache = DummyCache(read_wait_time=0.01, write_wait_time=0.0001)
    threads = [threading.Thread(target=thread_run_read) for _ in range(30)] + [
        threading.Thread(target=thread_run_write) for _ in range(30)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert cache.read(key=cache.DUMMY_KEY, wait_for_write_secs=0.01) == 99


def test_in_order_writes() -> None:
    # Test that the read- and write-through cache writes in-order.
    cache = DummyCache(read_wait_time=0.01, write_wait_time=0.0005)
    read_items: list[int] = []

    for idx in range(1000):
        cache.write(value=idx)

    for _ in range(1000):
        x = cache.read(key=cache.DUMMY_KEY, wait_for_write_secs=0.01)
        assert x is not None
        read_items.append(x)
        time.sleep(0.001)

    for idx in range(999):
        assert read_items[idx] <= read_items[idx + 1]
