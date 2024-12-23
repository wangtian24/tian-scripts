import asyncio
import inspect
import re
import time
from collections.abc import Callable, Generator
from threading import Lock
from typing import Any, Self, TypeVar, Union, no_type_check

import numpy as np


class SingletonMixin:
    _instance: Self | None = None
    _instance_lock = Lock()

    @classmethod
    def get_instance(cls, *args: Any, **kwargs: Any) -> Self:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(*args, **kwargs)

        return cls._instance  # type: ignore[return-value]


class RNGMixin:
    """
    Mixin class to add a random number generator to a class.
    """

    _rng: np.random.RandomState | None = None
    _seed: int | None = None
    _lock: Lock = Lock()

    def set_seed(self, seed: int, overwrite_existing: bool = False) -> None:
        with self._lock:
            if overwrite_existing:
                self._seed = None
                self._rng = None

            if self._seed is not None:
                raise ValueError("Seed already set")

            self._seed = seed
            self._rng = np.random.RandomState(self._seed)

    def with_seed(self, seed: int) -> Self:
        self.set_seed(seed)
        return self

    def get_seed(self) -> int:
        if self._seed is None:
            raise ValueError("Seed not set")

        return self._seed

    def get_rng(self) -> np.random.RandomState:
        with self._lock:
            if self._rng is None:
                self._rng = np.random.RandomState(self._seed)

        return self._rng


K = TypeVar("K")
V = TypeVar("V")


def dict_extract(d: dict[K, V], keys: set[K]) -> dict[K, V]:
    """Extracts a dictionary containing only the keys in `keys` from the dictionary `d`."""
    return {k: v for k, v in d.items() if k in keys}


FuncType = TypeVar("FuncType")


@no_type_check
def async_timed_cache(*, seconds: int, maxsize: int = 128) -> Callable[[FuncType], FuncType]:
    """
    Decorator to cache the result of a function for a given number of seconds. Each argument must be hashable. This is
    an LRU cache.
    """

    def decorator(func: FuncType) -> FuncType:
        async def wrapper(*args: Any, **kwargs: Any) -> V:
            key = (args, frozenset(kwargs.items()))
            now = time.time()

            if key in wrapper.__cache_kv:
                if now - wrapper.__cache_kv[key][1] > seconds:
                    wrapper.__cache_kv.pop(key)
                else:
                    wrapper.__cache_kv[key] = wrapper.__cache_kv.pop(key)

            if key not in wrapper.__cache_kv:
                if len(wrapper.__cache_kv) >= maxsize:
                    min_key = next(iter(wrapper.__cache_kv))
                    del wrapper.__cache_kv[min_key]

                wrapper.__cache_kv[key] = (await func(*args, **kwargs), now)

            return wrapper.__cache_kv[key][0]

        # values are tuples of (value, timestamp updated)
        # As of 3.6, dicts are now ordered
        wrapper.__cache_kv = {}

        return wrapper

    return decorator


T = TypeVar("T")
Result = Union[T, Exception]  # noqa


class EarlyTerminatedException(Exception):
    pass


class Delegator:
    def __init__(
        self,
        delegates: dict[str, Any],
        timeout_secs: float | None = None,
        return_when: str = asyncio.ALL_COMPLETED,
    ) -> None:
        """
        Creates a delegator that fans out method calls to underlying named objects.

        Args:
            delegates: Dictionary mapping names to delegate objects
            timeout: Timeout for method calls
            return_when: Whether to return the first result or all results
        """
        self.delegates = delegates
        self.timeout_secs = timeout_secs
        if return_when not in (asyncio.FIRST_COMPLETED, asyncio.ALL_COMPLETED):
            raise ValueError(f"Invalid return_when value: {return_when}")
        self.return_when = return_when

    async def delegate(self, method_name: str, *args: Any, **kwargs: Any) -> dict[str, Result]:
        """
        Delegates method calls to underlying objects, with optional timeout.

        Args:
            method_name: Name of the method to call on delegates
            *args, **kwargs: Arguments to pass to delegate methods

        Returns:
            Dictionary mapping delegate names to their results or exceptions
        """

        async def execute_method(name: str, delegate: Any) -> tuple[str, Result]:
            """Returns the name of the delegate and the result of the method call."""
            try:
                method = getattr(delegate, method_name)
                if inspect.iscoroutinefunction(method):
                    coro = method(*args, **kwargs)
                else:
                    coro = asyncio.to_thread(method, *args, **kwargs)

                result = await asyncio.wait_for(coro, self.timeout_secs)
                return name, result
            except asyncio.CancelledError:
                # Some other task cancelled this one
                return name, EarlyTerminatedException()
            except Exception as e:
                # Timeout, or underlying method actually raised
                return name, e

        if self.return_when == asyncio.ALL_COMPLETED:
            tasks = [execute_method(name, delegate) for name, delegate in self.delegates.items()]
            task_results = await asyncio.gather(*tasks)
            return dict(task_results)

        # Otherwise, wait for the first result and then cancel the rest
        pending = set()
        results = {}

        for name, delegate in self.delegates.items():
            task = asyncio.create_task(execute_method(name, delegate))
            pending.add(task)
        try:
            done, pending = await asyncio.wait(pending, return_when=self.return_when)

            first_task = done.pop()
            name, result = await first_task
            results[name] = result

            # Kill remaining tasks and wait for them to complete
            if pending:
                for task in pending:
                    task.cancel()
                done, _ = await asyncio.wait(pending)
                for task in done:
                    name, result = await task
                    results[name] = result

        except Exception as e:
            # Handle any unexpected errors
            for task in pending:
                if not task.done():
                    task.cancel()
            results.update({name: e for name in self.delegates.keys() if name not in results})

        # Assume everything else timed out
        results.update({name: TimeoutError() for name in self.delegates if name not in results})

        return results

    def __getattr__(self, name: str) -> Any:
        """Delegate any attribute access to the underlying delegates."""

        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Result]:
            return await self.delegate(name, *args, **kwargs)

        return wrapper


SPACE_REGEX = re.compile(r"\s+")
PUNCT_REGEX = re.compile(r"(\*|:|\.|,|!|;|\?|&|#|%|@|~|`|\(|\)|\"|')")


def simple_strip(text: str) -> str:
    return SPACE_REGEX.sub(" ", PUNCT_REGEX.sub("", text.lower().strip()))


MARKDOWN_REGEX = re.compile(r"(^|\n+)\s*(\*\*|\*\s|-|[0-9]+\.)(?P<content>[^\n]*)(\n|$)", re.MULTILINE)


def split_markdown_list(text: str, max_length: int = 500) -> Generator[tuple[int, int], None, None]:
    for match in MARKDOWN_REGEX.finditer(text):
        a, b = match.start("content"), match.end("content")

        if b - a > max_length:
            continue

        yield a, b
