from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import update
from sqlalchemy.engine import Result
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select
from ypl.backend.db import get_engine
from ypl.db.language_models import LanguageModel


class LanguageModelResponseBody(BaseModel):
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


def create_model(model: LanguageModel) -> UUID:
    with Session(get_engine()) as session:
        session.add(model)
        session.commit()
        session.refresh(model)
        return model.language_model_id


def get_available_models() -> list[LanguageModelResponseBody]:
    with Session(get_engine()) as session:
        query = (
            select(LanguageModel)
            .options(selectinload(LanguageModel.organization))  # type: ignore
            .where(LanguageModel.deleted_at.is_(None))  # type: ignore
        )
        models = session.exec(query).all()
        return [
            LanguageModelResponseBody(
                **model.model_dump(exclude={"organization"}),
                organization_name=model.organization.organization_name if model.organization else None,
            )
            for model in models
        ]


def get_model_details(model_id: str) -> LanguageModelResponseBody | None:
    with Session(get_engine()) as session:
        query = (
            select(LanguageModel)
            .options(selectinload(LanguageModel.organization))  # type: ignore
            .where(LanguageModel.language_model_id == model_id, LanguageModel.deleted_at.is_(None))  # type: ignore
        )
        model = session.exec(query).first()
        if model is None:
            return None
        return LanguageModelResponseBody(
            **model.model_dump(exclude={"organization"}),
            organization_name=model.organization.organization_name if model.organization else None,
        )


def update_model(model_id: str, updated_model: LanguageModel) -> LanguageModelResponseBody:
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

        return LanguageModelResponseBody(
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
