"""Create a cashout capability entry.

This is a one-off script to create a capability entry for cashout functionality
with appropriate configuration and restrictions.
"""

from datetime import UTC, datetime

from sqlalchemy import Connection
from sqlmodel import Session

from ypl.db.users import SYSTEM_USER_ID, Capability, CapabilityStatus


def create_cashout_capability(connection: Connection) -> None:
    """Create a cashout capability entry if it doesn't exist."""
    with Session(connection) as session:
        existing_capability = (
            session.query(Capability)
            .filter(
                Capability.capability_name == "cashout",
                Capability.deleted_at.is_(None),
                Capability.status != CapabilityStatus.DEPRECATED,
            )
            .first()
        )

        if existing_capability:
            return

        capability = Capability(
            capability_name="cashout",
            description="Capability to control access to cashout functionality",
            status=CapabilityStatus.ACTIVE,
            creator_user_id=SYSTEM_USER_ID,
            version_number=1,
            effective_date=datetime.now(UTC),
        )

        session.add(capability)
        session.commit()


def remove_cashout_capability(connection: Connection) -> None:
    """Soft delete the cashout capability entry by setting deleted_at."""
    with Session(connection) as session:
        capability = (
            session.query(Capability)
            .filter(
                Capability.capability_name == "cashout",
                Capability.deleted_at.is_(None),
            )
            .first()
        )

        if capability:
            capability.deleted_at = datetime.now(UTC)
            session.commit()
