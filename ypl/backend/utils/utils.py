import logging
import math
import time
import uuid
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any, TypedDict, cast
from urllib.parse import urlparse, urlunparse

from sqlalchemy import func, or_
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from ypl.backend.db import get_async_engine
from ypl.backend.utils.monitoring import metric_record
from ypl.db.users import (
    Capability,
    CapabilityStatus,
    User,
    UserCapabilityOverride,
    UserCapabilityStatus,
    WaitlistedUser,
)

UNKNOWN_USER = "Unknown User"
WAITLISTED_SUFFIX = " (**Waitlisted**)"


async def fetch_user_names(user_ids: list[str]) -> dict[str, str]:
    """Fetch multiple user names from the database in a single query."""
    try:
        engine = get_async_engine()
        async with AsyncSession(engine) as session:
            # First try to get names from users table
            query = select(User).where(
                User.user_id.in_(user_ids),  # type: ignore
            )
            users = (await session.exec(query)).all()

            name_dict = {user_id: user_id for user_id in user_ids}
            name_dict.update({user.user_id: str(user.name) for user in users if user.name})

            # For any remaining user_ids without names, check waitlisted_users table
            remaining_ids = [uid for uid in user_ids if name_dict[uid] == uid]
            if remaining_ids:
                waitlist_query = select(WaitlistedUser).where(
                    WaitlistedUser.waitlisted_user_id.in_([cast(uuid.UUID, uid) for uid in remaining_ids]),  # type: ignore
                )
                waitlisted_users = (await session.exec(waitlist_query)).all()

                # Update names from waitlisted users if found, adding the waitlisted suffix
                name_dict.update(
                    {
                        str(waitlisted_user.waitlisted_user_id): WAITLISTED_SUFFIX + str(waitlisted_user.name)
                        for waitlisted_user in waitlisted_users
                        if waitlisted_user.name
                    }
                )

            return name_dict

    except Exception as e:
        logging.exception(f"Failed to fetch users from database: {e}")
        return {user_id: user_id for user_id in user_ids}


async def fetch_user_name(user_id: str) -> str:
    """Fetch a single user name from the database."""
    try:
        engine = get_async_engine()
        async with AsyncSession(engine) as session:
            # First try users table
            query = select(User).where(
                User.user_id == user_id,
            )
            user = (await session.exec(query)).first()

            if user and user.name:
                return str(user.name)

            # If not found or no name, check waitlisted_users table
            waitlist_query = select(WaitlistedUser).where(
                WaitlistedUser.waitlisted_user_id == cast(uuid.UUID, user_id),
            )
            waitlisted_user = (await session.exec(waitlist_query)).first()

            if waitlisted_user and waitlisted_user.name:
                return WAITLISTED_SUFFIX + str(waitlisted_user.name)

            return UNKNOWN_USER

    except Exception as e:
        logging.exception(f"Failed to fetch user name from database: {e}")
        return UNKNOWN_USER


class CapabilityType(str, Enum):
    """Enum representing different capability types in the system."""

    CASHOUT = "cashout"


class CapabilityOverrideDetails(TypedDict):
    status: UserCapabilityStatus
    override_config: dict | None


async def get_capability_override_details(
    session: AsyncSession, user_id: str, capability_name: CapabilityType
) -> CapabilityOverrideDetails | None:
    """Check if a user has an entry in the user capability override table for the given capability type.
    If no entry exists, return None.
    If an entry exists and is disabled, indicate that the user is not eligible for this capability.
    If enabled, return the override configurations.
    """
    now = func.now()

    capability_stmt = select(Capability).where(
        Capability.capability_name == capability_name.value,
        Capability.deleted_at.is_(None),  # type: ignore
        Capability.status == CapabilityStatus.ACTIVE,
    )
    capability = (await session.exec(capability_stmt)).first()

    if not capability:
        return None

    # Then check for any active overrides
    override_stmt = select(UserCapabilityOverride).where(
        UserCapabilityOverride.user_id == user_id,
        UserCapabilityOverride.capability_id == capability.capability_id,
        UserCapabilityOverride.deleted_at.is_(None),  # type: ignore
        or_(
            UserCapabilityOverride.effective_start_date.is_(None),  # type: ignore
            UserCapabilityOverride.effective_start_date <= now,
        ),
        or_(
            UserCapabilityOverride.effective_end_date.is_(None),  # type: ignore
            UserCapabilityOverride.effective_end_date > now,
        ),
    )
    override = (await session.exec(override_stmt)).first()

    if not override:
        return None

    return {"status": override.status, "override_config": override.override_config}


class StopWatch:
    """
    StopWatch is a utility class for recording the time taken to execute a block of code.
    You can use it as a context manager or manually record splits.

        with StopWatch("latency/my_function"):
        x = my_function()

    Or use it manually, with a few splits in between:

        stopwatch = StopWatch()
        doStepA()
        stopwatch.record_split("step_a")
        doStepB()
        stopwatch.record_split("step_b")
        doStepC()
        stopwatch.end("step_c")
        stopwatch.export_metrics("latency/")  # split names will be appended to the prefix when exporting metrics

    You can also use it record laps with a start and end point you care about, rather than calculating from last split.

        stopwatch = StopWatch("latency/", auto_export=True)  # will export metrics when .end() is called
        doStuff()
        stopwatch.start_lap("core_step")
        doCoreStep()
        stopwatch.end_lap("core_step")
        doMoreStuff()
        stopwatch.end("more_stuff")
    """

    TOTAL_KEY = "TOTAL"

    def __init__(self, name: str | None = None, auto_export: bool = False) -> None:
        self.name = name
        self.split_start_time = time.time() * 1000
        self.stopwatch_start_time = self.split_start_time  # the overall start time
        self.splits: dict[str, int] = {}
        self.lap_starts: dict[str, float] = {}
        self.ended = False
        self.auto_export = auto_export

    def __enter__(self) -> "StopWatch":
        # nothing really to do here
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end()
        if self.auto_export:
            self.export_metrics()
        # self.pretty_print()  # uncomment this line for debugging purposes

    # Recording time for specific laps with its own start and end

    def record_split(self, split_name: str) -> None:
        """Record time since last split and update start time."""
        current = time.time() * 1000
        self.splits[split_name] = int(current - self.split_start_time)
        self.split_start_time = current

    def end(self, last_split_name: str | None = None) -> None:
        """Record final split and total time."""
        if not self.ended:
            if last_split_name:
                self.record_split(last_split_name)
            self.splits[self.TOTAL_KEY] = int(time.time() * 1000 - self.stopwatch_start_time)
            self.ended = True
            if self.auto_export:
                self.export_metrics()

    # Recording time for specific laps with its own start and end

    def start_lap(self, lap_name: str) -> None:
        """Start a new lap with the given name."""
        self.lap_starts[lap_name] = time.time() * 1000

    def end_lap(self, lap_name: str) -> None:
        """End a lap and record its duration."""
        if lap_name not in self.lap_starts:
            raise ValueError(f"No lap named '{lap_name}' was started")
        current = time.time() * 1000
        self.splits[lap_name] = int(current - self.lap_starts[lap_name])
        del self.lap_starts[lap_name]

    # Getting results

    def get_total_time(self) -> int:
        """Return total time in milliseconds."""
        if not self.ended:
            return int(time.time() * 1000 - self.stopwatch_start_time)
        return self.splits[self.TOTAL_KEY]

    def get_splits(self) -> dict[str, int]:
        """Return a copy of thedictionary of all recorded splits."""
        return self.splits.copy()

    def pretty_print(self, print_total: bool = True) -> None:
        """Print splits one per row with millisecond suffix."""
        print(f"StopWatch results: {self.name}")
        for split_name, duration in self.splits.items():
            if not print_total and split_name == self.TOTAL_KEY:
                continue
            print(f"-- {split_name:40} {int(duration):8} ms")

    def export_metrics(self, prefix: str | None = None) -> None:
        """Export all splits as metrics with the given prefix."""
        if prefix is None:
            prefix = self.name or ""

        for split_name, duration in self.splits.items():
            metric_record(f"{prefix}{split_name}_ms", duration)


async def yield_all(async_iter: AsyncIterator[Any]) -> AsyncIterator[Any]:
    async for msg in async_iter:
        yield msg


def merge_base_url_with_port(base_url: str, port: int | None) -> str:
    """Merge a base URL with a port number"""
    if not port:
        return base_url
    parsed_url = urlparse(base_url)
    netloc = f"{parsed_url.hostname}:{port}"
    return urlunparse(
        (parsed_url.scheme, netloc, parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment)
    )


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lat1: Latitude of first point in decimal degrees
        lon1: Longitude of first point in decimal degrees
        lat2: Latitude of second point in decimal degrees
        lon2: Longitude of second point in decimal degrees

    Returns:
        Distance in kilometers between the two points
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    return c * r
