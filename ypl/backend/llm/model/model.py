from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import String, cast, func, update
from sqlalchemy.engine import Result
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select
from ypl.backend.db import get_engine
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, LicenseEnum


class LanguageModelStruct(BaseModel):
    language_model_id: UUID
    name: str
    internal_name: str
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


def create_model(model: LanguageModel) -> UUID:
    model.status = LanguageModelStatusEnum.SUBMITTED
    with Session(get_engine()) as session:
        session.add(model)
        session.commit()
        session.refresh(model)
        return model.language_model_id


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
