from datetime import datetime
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import func, update
from sqlalchemy.engine import Result
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from ypl.backend.db import get_async_engine
from ypl.db.language_models import Provider


class ProviderStruct(BaseModel):
    provider_id: UUID
    name: str
    base_api_url: str
    is_active: bool


async def create_provider(provider: Provider) -> UUID:
    async with AsyncSession(get_async_engine()) as session:
        session.add(provider)
        await session.commit()
        await session.refresh(provider)
        return provider.provider_id


async def get_providers(
    name: str | None = None,
    is_active: bool | None = None,
    exclude_deleted: bool = True,
) -> list[ProviderStruct]:
    """Get providers from the database.

    Args:
        name: The name of the provider to filter by.
        is_active: The statuses of the provider to filter by.
        exclude_deleted: Whether to exclude deleted providers.
    Returns:
        The providers that match the filter criteria.
    """
    async with AsyncSession(get_async_engine()) as session:
        query = select(Provider)

        if name:
            query = query.where(func.lower(Provider.name) == name.lower())
        if exclude_deleted:
            query = query.where(Provider.deleted_at.is_(None))  # type: ignore
        if is_active is not None:
            query = query.where(Provider.is_active == is_active)

        result = await session.execute(query)
        providers = result.scalars().all()
        return [ProviderStruct(**provider.model_dump()) for provider in providers]


async def get_provider_details(provider_id: UUID) -> ProviderStruct | None:
    async with AsyncSession(get_async_engine()) as session:
        query = select(Provider).where(
            Provider.provider_id == provider_id,
            Provider.deleted_at.is_(None),  # type: ignore
        )
        result = await session.execute(query)
        provider = result.scalar_one_or_none()
        if provider is None:
            return None
        return ProviderStruct(**provider.model_dump())


async def update_provider(provider_id: UUID, updated_provider: Provider) -> ProviderStruct:
    async with AsyncSession(get_async_engine()) as session:
        provider_data = updated_provider.model_dump(exclude_unset=True, exclude={"provider_id"})
        if not provider_data:
            raise ValueError("No fields to update")

        existing_provider = await session.get(Provider, provider_id)
        if existing_provider is None:
            raise ValueError(f"Provider with id {provider_id} not found")

        for field, value in provider_data.items():
            setattr(existing_provider, field, value)

        existing_provider.modified_at = datetime.utcnow()

        await session.commit()
        await session.refresh(existing_provider)

        return ProviderStruct(**existing_provider.model_dump())


async def delete_provider(provider_id: UUID) -> None:
    async with AsyncSession(get_async_engine()) as session:
        result: Result = await session.execute(
            update(Provider)
            .where(Provider.provider_id == provider_id)  # type: ignore
            .values(deleted_at=datetime.utcnow())
        )

        if result.rowcount == 0:  # type: ignore
            raise ValueError(f"Provider with id {provider_id} not found")
        await session.commit()
