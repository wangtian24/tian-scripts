import asyncio
import logging
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def handle_task_exception(task: asyncio.Task) -> None:
    """Default exception handler for background tasks."""
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task was cancelled, this is normal
    except Exception as e:
        logging.error(f"Background task failed: {e}", exc_info=True)


def create_background_task(
    coroutine: Coroutine[Any, Any, T],
    *,
    exception_handler: Callable[[asyncio.Task], None] = handle_task_exception,
) -> asyncio.Task[T]:
    """
    Creates and schedules a background task with proper exception handling.

    Args:
        coroutine: The coroutine to run as a background task
        exception_handler: Optional custom exception handler function

    Returns:
        The created task
    """
    task: asyncio.Task[T] = asyncio.create_task(coroutine)
    task.add_done_callback(exception_handler)
    return task


def background_task(
    exception_handler: Callable[[asyncio.Task], None] = handle_task_exception,
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., asyncio.Task[T]]]:
    """
    Decorator to automatically convert a coroutine into a background task.

    Args:
        exception_handler: Optional custom exception handler function

    Returns:
        Decorator function that wraps the coroutine
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., asyncio.Task[T]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> asyncio.Task[T]:
            return create_background_task(func(*args, **kwargs), exception_handler=exception_handler)

        return wrapper

    return decorator
