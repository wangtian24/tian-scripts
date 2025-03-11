# Standard library imports
import logging
import os

from sqlalchemy import func, text
from sqlalchemy.orm import selectinload
from sqlmodel import select
from ypl.backend.db import get_async_session
from ypl.backend.llm.model.model import create_model_canonical_name
from ypl.backend.llm.utils import YuppSlackApps, post_to_slack_channel
from ypl.backend.utils.json import json_dumps
from ypl.db.language_models import (
    LanguageModel,
    LanguageModelStatusEnum,
    LanguageModelTaxonomy,
)


async def _log_and_post(message: str, details: list[str]) -> None:
    env = f"{os.environ.get('ENVIRONMENT')}"
    dict = {"message": f"Model Maintenance: {message}", "details": details}
    logging.info(json_dumps(dict))

    details_list = [f"- {detail}\n" for detail in details]
    slack_msg = f":eyeglasses: *Model Metadata Maintenance* [{env}] - {message}\n{''.join(details_list)}"
    await post_to_slack_channel(slack_msg, "#alert-model-management", YuppSlackApps.MODEL_MANAGEMENT)


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
            canonical_name = create_model_canonical_name(model.provider.name, model.internal_name)
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


async def _check_models_with_no_taxonomy_info() -> None:
    async with get_async_session() as session:
        query = select(LanguageModel).where(
            LanguageModel.taxonomy_id.is_(None),  # type: ignore
            LanguageModel.deleted_at.is_(None),  # type: ignore
        )
        models_missing_taxo_id = (await session.exec(query)).all()
        if models_missing_taxo_id and len(models_missing_taxo_id) > 0:
            model_names = [model.name for model in models_missing_taxo_id]
            await _log_and_post(f"Found {len(model_names)} models missing taxonomy ID", model_names)


async def _check_taxo_label_duplicates() -> None:
    async with get_async_session() as session:
        query = (
            select(LanguageModelTaxonomy.taxo_label, func.count().label("num_pickables"))
            .where(LanguageModelTaxonomy.is_pickable.is_(True))  # type: ignore
            .group_by(LanguageModelTaxonomy.taxo_label)
            .having(func.count() > 1)
        )
        results = (await session.exec(query)).all()
        if results and len(results) > 0:
            dupes = [
                f"taxo_label [{row.taxo_label}] has {row.num_pickables} pickable taxonomy entries"  # type: ignore
                for row in results
            ]
            await _log_and_post(
                f"Found {len(dupes)} taxo_labels with multiple pickable entries, "
                f"they will show up as duplicates in the model picker.",
                dupes,
            )


async def _check_model_taxo_without_active_model_provider() -> None:
    async with get_async_session() as session:
        # OK this is too complicated with ORM, just do the raw SQL
        query = text(
            """
            select
                lmt.taxo_label,
                count(case when lm.status = 'ACTIVE' and lm.deleted_at is null then 1 end) as active_not_deleted_count
            from language_model_taxonomy lmt
                left join language_models lm on lm.taxonomy_id = lmt.language_model_taxonomy_id
            where lmt.taxo_label in (
                select taxo_label
                from language_model_taxonomy
                group by taxo_label
                having sum(case when is_pickable then 1 else 0 end) > 0
            )
            group by lmt.taxo_label
            having count(case when lm.status = 'ACTIVE' and lm.deleted_at is null then 1 end) = 0
            order by lmt.taxo_label
            """
        )
        results = (await session.execute(query)).all()
        if results and len(results) > 0:
            await _log_and_post(
                f"Found {len(results)} pickable model types without active provider",
                [model.taxo_label for model in results],
            )


async def _check_model_provider_without_pickable_model_taxo() -> None:
    async with get_async_session() as session:
        query = (
            select(LanguageModel)
            .join(LanguageModelTaxonomy)
            .where(
                LanguageModelTaxonomy.is_pickable.is_(False),  # type: ignore
                LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
                LanguageModel.deleted_at.isnot(None),  # type: ignore
            )
        )
        results = (await session.exec(query)).all()
        if results:
            await _log_and_post(
                f"Found {len(results)} active provided models without pickable model type",
                [f"{model.name}" for model in results],
            )


async def do_model_metadata_maintenance() -> None:
    """check for model metadata issues and notify on slack, fix automaticallyif possible."""
    await _check_model_provider_and_names()
    await _check_models_with_no_taxonomy_info()
    await _check_taxo_label_duplicates()
    await _check_model_taxo_without_active_model_provider()
    await _check_model_provider_without_pickable_model_taxo()
