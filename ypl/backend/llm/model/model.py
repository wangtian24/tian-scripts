import logging
from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import String, cast, func, update
from sqlalchemy.engine import Result
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select
from sqlmodel.ext.asyncio.session import AsyncSession
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ypl.backend.db import get_async_engine, get_engine
from ypl.backend.llm.routing.route_data_type import LanguageModelStatistics
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, LicenseEnum
from ypl.utils import async_timed_cache


class LanguageModelStruct(BaseModel):
    language_model_id: UUID
    name: str
    label: str | None
    license: str
    family: str | None
    avatar_url: str | None
    parameter_count: int | None
    context_window_tokens: int | None
    knowledge_cutoff_date: date | None
    organization_name: str | None
    status: str
    creator_user_id: str
    provider_id: UUID | None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
def create_model(model: LanguageModel) -> UUID:
    if not model.internal_name:
        model.internal_name = model.name
    model.status = LanguageModelStatusEnum.SUBMITTED
    with Session(get_engine()) as session:
        session.add(model)
        session.commit()
        session.refresh(model)
        return model.language_model_id


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
def get_models(
    name: str | None = None,
    licenses: list[LicenseEnum] | None = None,
    family: str | None = None,
    statuses: list[LanguageModelStatusEnum] | None = None,
    creator_user_id: str | None = None,
    exclude_deleted: bool = True,
) -> list[LanguageModelStruct]:
    """Get models from the database.

    Args:
        name: The name of the model to filter by.
        license: The licenses of the model to filter by.
        family: The family of the model to filter by.
        status: The statuses of the model to filter by.
        creator_user_id: The creator user id of the model to filter by.
        exclude_deleted: Whether to exclude deleted models.
    Returns:
        The models that match the filter criteria.
    """
    with Session(get_engine()) as session:
        query = (
            select(LanguageModel).options(selectinload(LanguageModel.organization))  # type: ignore
        )

        if name:
            query = query.where(func.lower(LanguageModel.name) == name.lower())
        if licenses:
            query = query.where(
                func.lower(cast(LanguageModel.license, String)).in_([license.value.lower() for license in licenses])
            )
        if family:
            query = query.where(func.lower(LanguageModel.family) == family.lower())
        if exclude_deleted:
            query = query.where(LanguageModel.deleted_at.is_(None))  # type: ignore
        if statuses:
            query = query.where(
                func.lower(cast(LanguageModel.status, String)).in_([status.value.lower() for status in statuses])
            )
        if creator_user_id:
            query = query.where(LanguageModel.creator_user_id == creator_user_id)

        models = session.exec(query).all()
        return [
            LanguageModelStruct(
                **model.model_dump(exclude={"organization"}),
                organization_name=model.organization.organization_name if model.organization else None,
            )
            for model in models
        ]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
# Only active and inactive models are returned as part of the model details as these
# have been onboarded and used for ranking in past
def get_model_details(model_id: str) -> LanguageModelStruct | None:
    with Session(get_engine()) as session:
        query = (
            select(LanguageModel)
            .options(selectinload(LanguageModel.organization))  # type: ignore
            .where(
                LanguageModel.language_model_id == model_id,
                LanguageModel.deleted_at.is_(None),  # type: ignore
                LanguageModel.status.in_(  # type: ignore
                    [LanguageModelStatusEnum.ACTIVE, LanguageModelStatusEnum.INACTIVE]
                ),
            )
        )
        model = session.exec(query).first()
        if model is None:
            return None
        return LanguageModelStruct(
            **model.model_dump(exclude={"organization"}),
            organization_name=model.organization.organization_name if model.organization else None,
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
def update_model(model_id: str, updated_model: LanguageModel) -> LanguageModelStruct:
    with Session(get_engine()) as session:
        model_data = updated_model.model_dump(exclude_unset=True, exclude={"language_model_id"})
        if not model_data:
            raise ValueError("No fields to update")

        existing_model = session.get(LanguageModel, model_id)
        if existing_model is None:
            raise ValueError(f"Model with id {model_id} not found")

        for field, value in model_data.items():
            setattr(existing_model, field, value)

        existing_model.modified_at = datetime.utcnow()

        session.commit()
        session.refresh(existing_model)

        return LanguageModelStruct(
            **existing_model.model_dump(exclude={"organization"}),
            organization_name=existing_model.organization.organization_name if existing_model.organization else None,
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
def delete_model(model_id: str) -> None:
    with Session(get_engine()) as session:
        result: Result = session.execute(
            update(LanguageModel)
            .where(LanguageModel.language_model_id == model_id)  # type: ignore
            .values(deleted_at=datetime.utcnow())
        )

        if result.rowcount == 0:  # type: ignore
            raise ValueError(f"Model with id {model_id} not found")
        session.commit()


@async_timed_cache(seconds=600)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def get_model_base_statistics(model_id: str) -> LanguageModelStatistics:
    async with AsyncSession(get_async_engine()) as session:
        llm = await session.get(LanguageModel, model_id)

        if llm is None:
            raise ValueError(f"Language model with id {model_id} not found")

        return LanguageModelStatistics(
            first_token_avg_latency_ms=llm.first_token_avg_latency_ms,
            first_token_p90_latency_ms=llm.first_token_p90_latency_ms,
            output_avg_tps=llm.output_avg_tps,
            output_p90_tps=llm.output_p90_tps,
        )


class ModelResponseTelemetry(BaseModel):
    """Telemetry data for a model response.

    Attributes:
        request_timestamp: Timestamp (in ms) when the request to the language model was created
        first_token_timestamp: Timestamp (in ms) when the first token was received for streaming model.
            Absent for non-streaming model.
        last_token_timestamp: Timestamp (in ms) when the last token was received for streaming model,
            or when the response was completed if it was not a streaming request.
        completion_tokens: Number of tokens returned in the response.
    """

    requestTimestamp: float
    firstTokenTimestamp: float | None = None
    lastTokenTimestamp: float | None = None
    completionTokens: int | None = None
    experiments: dict | None = None
    chunks_count: int | None = None
