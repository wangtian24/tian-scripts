import asyncio
import inspect
import re
import time
from collections.abc import Callable, Generator
from functools import lru_cache
from threading import Lock
from typing import Any, Literal, Self, TypeVar, Union, cast, no_type_check

import numpy as np
import tiktoken
from langchain_core.messages import BaseMessage, HumanMessage


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
        early_terminate_on: list[str] | None = None,
        priority_groups: list[list[str]] | None = None,
    ) -> None:
        """
        Creates a delegator that fans out method calls to underlying named objects.

        Args:
            delegates: Dictionary mapping names to delegate objects
            timeout: Timeout for method calls
            return_when: Whether to return the first result or all results
            early_terminate_on: List of delegate names that will trigger early termination
            priority_groups: groups of delegates for finer early termination control. You should only set
                either `early_terminate_on` or `priority_groups`, but not both.
        """
        self.delegates = delegates
        self.timeout_secs = timeout_secs
        assert not (
            early_terminate_on and priority_groups
        ), "Only one of early_terminate_on or priority_groups can be set"

        self.priority_groups = priority_groups or list([list(early_terminate_on)] if early_terminate_on else [[]])

        # do some sanity checks
        self.priority_groups_set = {name for group in self.priority_groups for name in group}
        missing_names = self.priority_groups_set - set(self.delegates.keys())
        if missing_names:
            raise ValueError(
                f"early_terminate_on or priority_groups contains names that are not in delegates: {missing_names}"
            )

    def _can_early_terminate(self, name: str, results: dict[str, Result]) -> bool:
        """
        We can early terminate and cancel all pending tasks if:
        1. this name is in the first priority group, OR
        2. all tasks in all higher priority group's delegates have failed.
        """
        if len(self.priority_groups_set) == 0:
            # No priority groups, no early termination.
            return False

        # Find which priority group this name belongs to
        current_group_idx = next(i for i, group in enumerate(self.priority_groups) if name in group)
        if current_group_idx == 0:
            # First priority group - can terminate early
            return True

        # Check if all results from higher priority groups failed
        all_failed = True  # whether all higher priority tasks have failed
        for group_idx in range(current_group_idx):
            for delegate_name in self.priority_groups[group_idx]:
                if delegate_name not in results.keys() or not isinstance(results[delegate_name], Exception):
                    all_failed = False
                    break
            if not all_failed:
                break
        return all_failed

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

        try:
            pending = set()
            results = {}

            # Create tasks for all delegates
            for name, delegate in self.delegates.items():
                task = asyncio.create_task(execute_method(name, delegate))
                pending.add(task)

            # Keep processing until all tasks complete or early termination
            while pending:
                # Wait for any tasks to complete
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                # Process completed tasks
                for task in done:
                    name, result = await task
                    results[name] = result
                    # Check if this result should trigger early termination
                    if (
                        name in self.priority_groups_set
                        and not isinstance(result, Exception)
                        and self._can_early_terminate(name, results)
                    ):
                        # Cancel remaining tasks
                        for task in pending:
                            task.cancel()
                        # Wait for cancellations to complete
                        if pending:
                            cancelled_done, pending = await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
                            # `pending` is now empty, as there is no timeout for wait().
                            for task in cancelled_done:
                                name, result = await task
                                results[name] = result
                        break

                    # TODO(Raghu): tweak: if all early_terminate_on models refuse, we wait for all of the remaining.
                    #              We only need one of the remaining to respond.

        except Exception as e:
            # Handle any unexpected errors
            for task in pending:
                if not task.done():
                    task.cancel()
            results.update({name: e for name in self.delegates.keys() if name not in results})

        # A catch-all for any unexpected errors that are not caught by execute_method()
        results.update({name: RuntimeError("Unexpected") for name in self.delegates if not results.get(name)})

        return results

    def __getattr__(self, name: str) -> Any:
        """Delegate any attribute access to the underlying delegates."""

        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Result]:
            return await self.delegate(name, *args, **kwargs)

        return wrapper


@lru_cache(maxsize=1000)
def compiled_regex(pattern: str, flags: int = 0) -> re.Pattern:
    return re.compile(pattern, flags)


SPACE_REGEX = re.compile(r"\s+")
PUNCT_REGEX = re.compile(r"(\*|:|\.|,|!|;|\?|&|#|%|@|~|`|\(|\)|\"|')")


MARKDOWN_REGEX = re.compile(r"(^|\n+)\s*(\*\*|\*\s|-|[0-9]+\.)(?P<content>[^\n]*)(\n|$)", re.MULTILINE)


def split_markdown_list(text: str, max_length: int = 500) -> Generator[tuple[int, int], None, None]:
    for match in MARKDOWN_REGEX.finditer(text):
        a, b = match.start("content"), match.end("content")

        if b - a > max_length:
            continue

        yield a, b


def tiktoken_trim(
    text: str, max_length: int, *, model: str = "gpt-4o", direction: Literal["left", "right"] = "left"
) -> str:
    enc = tiktoken.encoding_for_model(model)

    match direction:
        case "left":
            return enc.decode(enc.encode(text)[:max_length])
        case "right":
            return enc.decode(enc.encode(text)[-max_length:])
        case _:
            raise ValueError(f"Invalid direction: {direction}")


def get_text_part(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    text_parts = [part["text"] for part in message.content if isinstance(part, dict) and part.get("type") == "text"]
    return "\n".join(text_parts)


def replace_text_part(message: BaseMessage, new_text: str) -> HumanMessage:
    if isinstance(message.content, str):
        return HumanMessage(content=new_text)
    new_content: list[str | dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, str):
            new_content.append(part)
            continue
        part = cast(dict[str, Any], part)
        if part["type"] != "text":
            new_content.append(part)
            continue
        new_content.append({"type": "text", "text": part["text"] + "\n" + new_text})
    return HumanMessage(content=new_content)


def ifnull(value: Any, default_value: T) -> T:
    """
    Similar to ifnull() in SQL. Returns the value if it is not None, otherwise returns the default value.
    This is useful for avoiding type check errors while accessing ORM fields which are often defined as 'type | None'.
    """
    return default_value if value is None else value  # type: ignore[no-any-return]


def extract_json_dict_from_text(text: str) -> str:
    """
    Returns the substring in text between first '{' and last '}', including the braces.
    This does not validate if the returned string is valid json.
    This is often used to extract json response from an LLM. This response can contain extra text
    before and after the actual json.

    If the braces are not found or not in the correct order, it returns the original text.
    It does not raise an error.
    """

    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last == -1 or first > last:
        return text
    else:
        return text[first : last + 1]


_TRUNCATE_WITH = "... (truncated)"


def maybe_truncate(input: str, max_length: int) -> str:
    """Truncates the string with "... (truncated)' if it is longer than `max_length`."""
    if input and len(input) > max_length:
        assert max_length >= len(_TRUNCATE_WITH)
        return input[: (max_length - len(_TRUNCATE_WITH))] + _TRUNCATE_WITH
    return input
