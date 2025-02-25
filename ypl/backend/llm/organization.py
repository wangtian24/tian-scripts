from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import func, update
from sqlalchemy.engine import Result
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ypl.backend.db import get_async_engine
from ypl.db.language_models import Organization


class OrganizationStruct(BaseModel):
    organization_id: UUID
    name: str


async def create_organization(name: str) -> UUID:
    async with AsyncSession(get_async_engine()) as session:
        organization = Organization(organization_name=name)  # noqa: F821
        session.add(organization)
        await session.commit()
        await session.refresh(organization)
        return organization.organization_id


async def get_organizations(
    name: str | None = None,
    exclude_deleted: bool = True,
) -> list[OrganizationStruct]:
    """Get organizations from the database.

    Args:
        name: The name of the organization to filter by.
        exclude_deleted: Whether to exclude deleted organizations.
    Returns:
        The organizations that match the filter criteria.
    """
    async with AsyncSession(get_async_engine()) as session:
        query = select(Organization)

        if name:
            query = query.where(func.lower(Organization.organization_name) == name.lower())
        if exclude_deleted:
            query = query.where(Organization.deleted_at.is_(None))  # type: ignore

        result = await session.execute(query)
        organizations = result.scalars().all()
        return [
            OrganizationStruct(organization_id=org.organization_id, name=org.organization_name) for org in organizations
        ]


async def delete_organization(organization_id: UUID) -> None:
    async with AsyncSession(get_async_engine()) as session:
        result: Result = await session.execute(
            update(Organization)
            .where(Organization.organization_id == organization_id)  # type: ignore
            .values(deleted_at=func.now())
        )

        if result.rowcount == 0:  # type: ignore
            raise ValueError(f"Organization with id {organization_id} not found")
        await session.commit()
