import time
from collections.abc import Callable
from threading import Lock
from typing import Any, Self, TypeVar, no_type_check

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
