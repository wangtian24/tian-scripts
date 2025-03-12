import logging
import time
import uuid
from collections.abc import Hashable
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Condition, RLock
from typing import Any, Generic, TypeVar

from cachetools import LRUCache
from sqlmodel import Session, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.db import get_async_engine, get_engine
from ypl.db.chats import TurnQuality
from ypl.utils import SingletonMixin

KeyType = TypeVar("KeyType", bound=Hashable)
ValueType = TypeVar("ValueType")


class ReadWriteThroughCache(Generic[KeyType, ValueType]):
    """
    Defines a generic interface for a read- and write-through cache. It maintains a thread-safe in-memory cache
    of all the key-value pairs of `KeyType` and `ValueType` that have been transparently read- and written-through
    to the underlying storage, such as a database. The use case is for resource sharing between different parts of
    the backend that need access to the same data, and further operations on that data are expensive.

    Specifically, given a key, the cache will transparently read the value from the cache if it exists; otherwise,
    it waits for `wait_for_write_secs` seconds to wait for the value to be written by the asynchronous writer before
    performing the underlying read operation. If the key is supplied to the write operation, any subsequent read
    operations return immediately with the cached value. If the key is not supplied (e.g., when the key is not known
    to the writer ahead of time), the reader will block for a maximum of `wait_for_write_secs` seconds until the value
    is available. Read operations MAY block, and write operations NEVER block.

    For a concrete implementation, see :py:class:`.ThreadedReadWriteThroughCache`.

    See Also:
        - :py:class:`.ThreadedReadWriteThroughCache`
        - :py:class:`.DummyCache`
        - :py:module:`.tests.test_cache`
    """

    def __init__(self, *, maxsize: int = 4096):
        self._cache: dict[KeyType, ValueType] = LRUCache(maxsize=maxsize)  # type: ignore[assignment]
        self._cache_lock = RLock()

    def read(self, key: KeyType, *, wait_for_write_secs: float | None = 0.01, deep: bool = False) -> ValueType | None:
        """
        Reads the value from the cache, blocking for at most `wait_for_write_secs` seconds to wait for the value to
        be written by the asynchronous writer before performing the underlying read operation.

        Args:
            key: The key to read the value for.
            wait_for_write_secs: The maximum time to wait for the value to be written by the asynchronous writer.
            deep: If True, performs a deep read, i.e., reads the value from the underlying storage.
        """
        if deep:
            wait_for_write_secs = 0.0
        else:
            with self._cache_lock:
                if key in self._cache:
                    return self._cache[key]

        return self._read_impl(key, wait_for_write_secs=wait_for_write_secs)

    def _read_impl(self, key: KeyType, *, wait_for_write_secs: float | None = None) -> ValueType | None:
        """The blocking read operation for cache misses"""
        raise NotImplementedError

    def write(self, *, key: KeyType | None = None, value: ValueType) -> None:
        """Writes the value to the cache, _never_ blocking."""
        self._write_impl(key, value)

    def _write_impl(self, key: KeyType | None, value: ValueType) -> None:
        raise NotImplementedError

    async def aread(
        self, key: KeyType, *, wait_for_write_secs: float | None = 0.01, deep: bool = False
    ) -> ValueType | None:
        """Asynchronous counterpart to `read`."""
        if deep:
            wait_for_write_secs = 0.0
        else:
            with self._cache_lock:
                if key in self._cache:
                    return self._cache[key]

        value = await self._aread_impl(key, wait_for_write_secs=wait_for_write_secs)

        if value is not None:
            with self._cache_lock:
                self._cache[key] = value

        return value

    async def _aread_impl(self, key: KeyType, *, wait_for_write_secs: float | None = None) -> ValueType | None:
        """The blocking read operation for cache misses."""
        return self._read_impl(key, wait_for_write_secs=wait_for_write_secs)


class ThreadedReadWriteThroughCache(ReadWriteThroughCache[KeyType, ValueType]):
    """A read- and write-through cache that uses a thread pool for the write operations."""

    def __init__(self, *, max_workers: int = 10, maxsize: int = 4096):
        super().__init__(maxsize=maxsize)
        self.max_workers = max_workers
        self.write_executor = ThreadPoolExecutor(max_workers=max_workers)

        self._key_cache_lock = RLock()  # global lock for the key caches below

        # Key cache for the waiting for write condition variable
        self._key_to_waiting_for_write: dict[KeyType, Condition] = LRUCache(  # type: ignore[assignment]
            maxsize=maxsize
        )

        # Key cache for the has written flag, to be used with the waiting for write condition variable
        self._key_to_has_written: dict[KeyType, bool] = LRUCache(maxsize=maxsize)  # type: ignore[assignment]

        self._queue_lock = RLock()  # lock for processing the write queue
        self._write_queue: Queue[tuple[KeyType | None, ValueType]] = Queue()

    def _try_get_cached_value(self, key: KeyType, *, wait_for_write_secs: float | None = 0.01) -> ValueType | None:
        """
        Tries to get the cached value for the given key, blocking for at most `wait_for_write_secs` seconds if the
        value is not immediately available.
        """
        # Acquire lock to read and modify the key caches
        with self._key_cache_lock:  # acquire order: key_cache_lock
            # Initialize the key caches if they don't exist
            if key not in self._key_to_has_written:
                self._key_to_has_written[key] = False  # key has not been written yet
                self._key_to_waiting_for_write[key] = Condition()  # initialize the waiting for write CV

            # Try waiting for the value to be written to
            self._try_wait_for_write_secs(key, wait_for_write_secs)

            # Try returning the cached value
            return self._try_cache_fetch(key)

    def _try_cache_fetch(self, key: KeyType) -> ValueType | None:
        """Try returning the cached value."""
        with self._key_cache_lock:  # acquire order: key_cache_lock
            if key in self._key_to_has_written and self._key_to_has_written[key]:
                with self._cache_lock:  # acquire order: key_cache_lock -> cache_lock
                    if key in self._cache:
                        return self._cache[key]

        return None

    def _try_wait_for_write_secs(self, key: KeyType, wait_for_write_secs: float | None) -> None:
        """Try to wait for the value to be written by the writer."""
        # Acquire the waiting for write condition variable for the key
        with (cond := self._key_to_waiting_for_write[key]):  # acquire order: key_cache_lock -> cond
            # If it hasn't been written to, try waiting a bit for it to be written to
            if not self._key_to_has_written[key]:
                self._key_cache_lock.release()

                try:
                    cond.wait(timeout=wait_for_write_secs)
                finally:
                    self._key_cache_lock.acquire()

        # At this point, either the value has been written to, or the wait has timed out
        return

    def _read_impl(self, key: KeyType, *, wait_for_write_secs: float | None = 0.01) -> ValueType | None:
        if (
            wait_for_write_secs != 0.0
            and (value := self._try_get_cached_value(key, wait_for_write_secs=wait_for_write_secs)) is not None
        ):
            return value

        return self._full_read_impl(key)

    async def _aread_impl(self, key: KeyType, *, wait_for_write_secs: float | None = 0.01) -> ValueType | None:
        if (
            wait_for_write_secs != 0.0
            and (value := self._try_get_cached_value(key, wait_for_write_secs=wait_for_write_secs)) is not None
        ):
            return value

        return await self._afull_read_impl(key)

    def _full_read_impl(self, key: KeyType) -> ValueType | None:
        raise NotImplementedError

    async def _afull_read_impl(self, key: KeyType) -> ValueType | None:
        raise NotImplementedError

    def _write_impl(self, key: KeyType | None, value: ValueType) -> None:
        self._write_queue.put((key, value))
        self.write_executor.submit(self._th_base_write_impl)

    def _th_base_write_impl(self) -> None:
        with self._queue_lock:  # acquire order: queue_lock
            key, value = self._write_queue.get()

            if key is not None:  # key doesn't have to be computed by the write op, so add it to the cache
                self._add_to_cache(key, value)

            new_key = self._th_write_impl(key, value)  # do the actual write operation

            if key is None:  # key had to be computed by the write op (new_key is the key to add to the cache)
                self._add_to_cache(new_key, value)

    def _add_to_cache(self, key: KeyType, value: ValueType) -> None:
        with self._key_cache_lock:  # acquire order: key_cache_lock
            with self._cache_lock:  # acquire order: key_cache_lock -> cache_lock
                self._cache[key] = value

            if key not in self._key_to_waiting_for_write:
                self._key_to_waiting_for_write[key] = Condition()

            self._key_to_has_written[key] = True

            # acquire order: key_cache_lock -> cond
            with (cond := self._key_to_waiting_for_write[key]):
                cond.notify_all()

    def _th_write_impl(self, key: KeyType | None, value: ValueType) -> KeyType:
        """The actual write operation to be executed by the writer thread pool executor."""
        raise NotImplementedError


class DummyCache(ThreadedReadWriteThroughCache[str, int]):
    DUMMY_KEY = "dummy_key"
    full_read = False
    full_write = False

    def __init__(self, *, read_wait_time: float = 2, write_wait_time: float = 1, **kwargs: Any):
        super().__init__(**kwargs)
        self.read_wait_time = read_wait_time
        self.write_wait_time = write_wait_time

    def _full_read_impl(self, key: str) -> int | None:
        time.sleep(self.read_wait_time)
        self.full_read = True

        return 5

    def _th_write_impl(self, key: str | None, value: int) -> str:
        time.sleep(self.write_wait_time)
        self.full_write = True

        return self.DUMMY_KEY


class TurnQualityCache(SingletonMixin, ThreadedReadWriteThroughCache[uuid.UUID, TurnQuality]):
    """A cache for reading and writing turn qualities."""

    def _full_read_impl(self, key: uuid.UUID) -> TurnQuality | None:
        with Session(get_engine()) as session:
            try:
                return session.exec(select(TurnQuality).where(TurnQuality.turn_id == key)).first()
            except Exception as e:
                logging.exception(f"Error reading turn quality for {key}: {e}")
                return None

    async def _afull_read_impl(self, key: uuid.UUID) -> TurnQuality | None:
        async with AsyncSession(get_async_engine()) as session:
            try:
                tq = (await session.exec(select(TurnQuality).where(TurnQuality.turn_id == key))).first()
            except Exception as e:
                logging.exception(f"Error reading turn quality for {key}: {e}")
                return None

            if tq is not None:
                # Detach the turn
                session.expunge(tq)

            return tq

    def _th_write_impl(self, key: uuid.UUID | None, value: TurnQuality) -> uuid.UUID:
        with Session(get_engine()) as session:
            if key is None:
                raise ValueError("Key cannot be None")

            turn_quality = session.exec(select(TurnQuality).where(TurnQuality.turn_id == key)).first()

            if turn_quality is None:
                raise ValueError(f"Turn quality not found for {key}")
            else:
                turn_quality.prompt_difficulty = value.prompt_difficulty
                turn_quality.prompt_difficulty_judge_model_id = value.prompt_difficulty_judge_model_id
                turn_quality.prompt_difficulty_details = value.prompt_difficulty_details
                turn_quality.prompt_is_safe = value.prompt_is_safe
                turn_quality.prompt_unsafe_reasons = value.prompt_unsafe_reasons
                turn_quality.prompt_moderation_model_name = value.prompt_moderation_model_name
                turn_quality.is_suggested_followup = value.is_suggested_followup
                turn_quality.is_conversation_starter = value.is_conversation_starter

            session.add(turn_quality)
            session.commit()

        return turn_quality.turn_id
