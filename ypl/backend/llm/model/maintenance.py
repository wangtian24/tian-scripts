# Standard library imports
import logging
import os

from sqlalchemy.orm import selectinload
from sqlmodel import select
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.db.language_models import (
    LanguageModel,
)


async def _log_and_post(message: str, details: list[str]) -> None:
    env = f"{os.environ.get('ENVIRONMENT')}"
    dict = {"message": f"Model Maintenance: {message}", "details": details}
    logging.info(json_dumps(dict))

    details_list = [f"- {detail}\n" for detail in details]
    slack_msg = f":eyeglasses: *Model Metadata Maintenance* [{env}] - {message}\n{''.join(details_list)}"
    await post_to_slack(slack_msg)


async def _check_model_provider_and_names() -> None:
    async with get_async_session() as session:
        query = (
            select(LanguageModel)
            .options(selectinload(LanguageModel.provider))  # type: ignore
            .where(
                LanguageModel.deleted_at.is_(None),  # type: ignore
            )
        )

        result = await session.exec(query)
        models = result.all()

        models_with_name_changes: list[tuple[str, str]] = []
        models_without_providers: list[str] = []
        for model in models:
            if model.provider is None:
                models_without_providers.append(model.name)
                continue
            canonical_name = f"{model.provider.name.lower().replace(' ', '_')}/{model.internal_name}"
            if model.name != canonical_name:
                models_with_name_changes.append((model.name, canonical_name))
                model.name = canonical_name
                session.add(model)
        await session.commit()

    # report updated model names
    if len(models_without_providers) > 0:
        await _log_and_post(f"Found {len(models_without_providers)} models with no provider", models_without_providers)

    if len(models_with_name_changes) > 0:
        all_changes_str = [f"{before} -> {after}" for before, after in models_with_name_changes]
        await _log_and_post(f"Properly formatted names of {len(models_with_name_changes)} models", all_changes_str)


async def _find_models_with_no_taxonomy_info() -> None:
    async with get_async_session() as session:
        query = select(LanguageModel).where(
            LanguageModel.taxonomy_id.is_(None),  # type: ignore
            LanguageModel.deleted_at.is_(None),  # type: ignore
        )
        models_missing_taxo_id = await session.exec(query)
        if models_missing_taxo_id:
            model_names = [model.name for model in models_missing_taxo_id]
            await _log_and_post(f"Found {len(model_names)} models missing taxonomy ID", model_names)


async def do_model_metadata_maintenance() -> None:
    """check for model metadata issues and notify on slack, fix automaticallyif possible."""
    await _check_model_provider_and_names()
    await _find_models_with_no_taxonomy_info()
