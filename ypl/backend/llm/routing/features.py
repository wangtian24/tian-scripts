from pydantic import BaseModel
from sqlmodel import select

from ypl.backend.db import get_async_session
from ypl.backend.llm.attachment import supports_image, supports_pdf
from ypl.backend.llm.routing.common import SelectIntent
from ypl.db.language_models import LanguageModel, Provider
from ypl.utils import async_timed_cache

"""
Features for request context (user and prompt) and models
"""


class RequestContext(BaseModel):
    intent: SelectIntent
    user_required_models: list[str]
    inherited_models: list[str]
    prompt_categories: list[str]


class ModelFeatures(BaseModel):
    # will be filled by model_dump()
    is_pro: bool
    is_strong: bool
    is_fast: bool
    is_reasoning: bool
    is_image_generation: bool
    is_live: bool
    # will need to be calculate
    can_process_pdf: bool
    can_process_image: bool


class ModelSetFeatures(BaseModel):
    model_features: dict[str, ModelFeatures]  # map from model full name to model features


@async_timed_cache(seconds=600)
async def collect_model_features() -> ModelSetFeatures:
    """
    Collect model features from DB and return a ModelSetFeatures object.
    TODO(Tian): collect more features from various sources and use this for a ranking router.
    """
    async with get_async_session() as session:
        # here we are just collecting information, we don't need them to be active
        query = (
            select(LanguageModel, Provider)
            .join(Provider)
            .where(
                LanguageModel.deleted_at.is_(None),  # type: ignore
                Provider.deleted_at.is_(None),  # type: ignore
            )
        )
        models = await session.exec(query)
        model_features = {
            model.internal_name: ModelFeatures(
                **model.model_dump(
                    exclude={"is_pro", "is_strong", "is_fast", "is_reasoning", "is_image_generation", "is_live"}
                ),  # extract fields with the same name
                is_pro=model.is_pro or False,
                is_strong=model.is_strong or False,
                is_fast=provider.is_fast or False,
                is_reasoning=model.is_reasoning or False,
                is_image_generation=model.is_image_generation or False,
                is_live=model.is_live or False,
                can_process_pdf=(
                    model.supported_attachment_mime_types is not None
                    and supports_pdf(model.supported_attachment_mime_types)
                ),
                can_process_image=(
                    model.supported_attachment_mime_types is not None
                    and supports_image(model.supported_attachment_mime_types)
                ),
            )
            for model, provider in models
        }
        return ModelSetFeatures(model_features=model_features)
