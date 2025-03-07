import logging
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import String, cast, func, update
from sqlalchemy.engine import Result
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.orm import selectinload
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ypl.backend.db import get_async_session
from ypl.backend.llm.attachment import supports_image, supports_pdf
from ypl.backend.llm.routing.route_data_type import LanguageModelStatistics
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, LanguageModelTaxonomy, LicenseEnum, Provider
from ypl.utils import async_timed_cache


class LanguageModelStruct(BaseModel):
    created_at: datetime
    language_model_id: UUID
    name: str
    internal_name: str
    label: str | None
    license: str
    model_publisher: str | None
    model_family: str | None
    model_class: str | None
    model_version: str | None
    model_release: str | None
    avatar_url: str | None
    parameter_count: int | None
    context_window_tokens: int | None
    knowledge_cutoff_date: date | None
    organization_name: str | None
    status: str
    creator_user_id: str
    provider_id: UUID | None
    parameters: dict[str, Any] | None
    supported_attachment_mime_types: list[str] | None
    external_model_info_url: str | None
    # below are inferred from elsewhere and not directly dumped from the LanguageModel fields
    provider_name: str | None
    taxonomy_id: UUID | None
    taxonomy_label: str | None
    taxonomy_path: str | None  # joining publisher, family, class, version, release
    flags: list[str] | None  # a list of PRO, STRONG, LIVE, FAST, etc
    is_internal: bool | None
    keywords: list[str] | None  # a list of keywords the client can match on in typeahead
    priority: int | None


def clean_provider_name(provider_name: str) -> str:
    return provider_name.lower().strip().replace(" ", "_")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def create_model(model: LanguageModel) -> UUID:
    # TODO(Tian): current UI just asks for 'name', we should ask for 'internal_name'. The 'name' will be set
    # automatically later using the provider name and internal name.
    if not model.internal_name:
        model.internal_name = model.name

    async with get_async_session() as session:
        # Set model's name to "provider_name/model_name" if provider_id is set
        if model.provider_id:
            provider_query = select(Provider.name).where(Provider.provider_id == model.provider_id)
            provider_name = (await session.exec(provider_query)).one_or_none()
            await session.commit()
            if provider_name:
                model.name = f"{clean_provider_name(provider_name)}/{model.internal_name}"
            else:
                raise ValueError(f"Provider {model.provider_id} not found")

        # Check if a taxonomy id exists, if so, link to it as well
        if model.family and (model.model_class or model.model_version or model.model_release):
            taxo_id = await _find_taxonomy(
                session, model.family, model.model_class, model.model_version, model.model_release
            )
            if taxo_id:
                model.taxonomy_id = taxo_id
            else:
                model.taxonomy_id = await _create_model_taxonomy_from_model(model)

        # add model
        session.add(model)
        await session.commit()
        return model.language_model_id


async def _find_taxonomy(
    session: AsyncSession, family: str, class_: str | None, version: str | None, release: str | None
) -> UUID | None:
    # Check if taxonomy with same attributes already exists
    query = select(LanguageModelTaxonomy.language_model_taxonomy_id).where(LanguageModelTaxonomy.model_family == family)

    # Note that if a field is None, we only match on null/empty string in DB, rather than
    # ignore the field.
    if class_:
        query = query.where(LanguageModelTaxonomy.model_class == class_)
    else:
        query = query.where(LanguageModelTaxonomy.model_class.is_(None) | (LanguageModelTaxonomy.model_class == ""))  # type: ignore

    if version:
        query = query.where(LanguageModelTaxonomy.model_version == version)
    else:
        query = query.where(
            LanguageModelTaxonomy.model_version.is_(None) | (LanguageModelTaxonomy.model_version == "")  # type: ignore
        )

    if release:
        query = query.where(LanguageModelTaxonomy.model_release == release)
    else:
        query = query.where(
            LanguageModelTaxonomy.model_release.is_(None) | (LanguageModelTaxonomy.model_release == "")  # type: ignore
        )

    taxo_id = (await session.exec(query)).one_or_none()
    await session.commit()
    return taxo_id


async def _create_model_taxonomy_from_model(model: LanguageModel) -> UUID:
    taxo = LanguageModelTaxonomy(
        taxo_label=model.label,
        model_publisher="UNKNOWN",  # not yet on UI nor a field in model, will modify later
        model_family=model.family,
        model_class=model.model_class,  # not yet on UI
        model_version=model.model_version,  # not yet on UI
        model_release=model.model_release,  # not yet on UI
        parameter_count=model.parameter_count,  # not yet on UI
        context_window_tokens=model.context_window_tokens or None,
        knowledge_cutoff_date=model.knowledge_cutoff_date,
        supported_model_ids=model.supported_attachment_mime_types,  # not yet on UI
        avatar_url=model.avatar_url,
        is_strong_model=model.is_strong,  # not yet on UI
        is_pro=model.is_pro,  # not yet on UI
        is_live=model.is_live,  # not yet on UI
    )
    async with get_async_session() as session:
        session.add(taxo)
        await session.commit()
        return taxo.language_model_taxonomy_id


async def create_model_taxonomy(taxo: LanguageModelTaxonomy) -> UUID:
    """
    Create a new taxonomy and return its id, if already exist, return existing id.
    """
    async with get_async_session() as session:
        # Check if taxonomy with same attributes already exists
        assert taxo.model_family is not None
        existing_taxo_id = await _find_taxonomy(
            session, taxo.model_family, taxo.model_class, taxo.model_version, taxo.model_release
        )
        if existing_taxo_id:
            return existing_taxo_id

        session.add(taxo)
        await session.commit()
        return taxo.language_model_taxonomy_id


class ModelTaxonomyQuery(BaseModel):
    taxonomy_id: UUID | None = None
    model_publisher: Optional[str] | None = None  # noqa: UP007
    model_family: Optional[str] | None = None  # noqa: UP007
    model_class: Optional[str] | None = None  # noqa: UP007
    model_version: Optional[str] | None = None  # noqa: UP007
    model_release: Optional[str] | None = None  # noqa: UP007
    pickable_only: bool | None = True
    leaf_node_only: bool | None = False
    has_active_providers: bool | None = True


class ModelTaxonomyResponse(BaseModel):
    created_at: datetime
    taxonomy_id: UUID
    taxonomy_label: str
    taxonomy_path: str | None  # joining publisher, family, class, version, release
    model_publisher: str
    model_family: str
    model_class: str | None
    model_version: str | None
    model_release: str | None
    # other metadata
    avatar_url: str | None
    parameter_count: int | None
    context_window_tokens: int | None
    knowledge_cutoff_date: date | None
    supported_attachment_mime_types: list[str] | None
    # below are inferred from elsewhere and not directly dumped from the LanguageModel fields
    flags: list[str] | None  # a list of PRO, STRONG, LIVE, FAST, etc
    is_internal: bool | None
    keywords: list[str] | None  # a list of keywords the client can match on in typeahead
    priority: int | None


MAX_NEW_MODEL_AGE_DAYS = 30


def _is_none_or_null(value: str) -> bool:
    return value.lower() in ["none", "null"]


def _make_taxo_path_from_taxonomy(taxo: LanguageModelTaxonomy | None) -> str:
    return _make_taxo_path(
        taxo.model_publisher if taxo else None,
        taxo.model_family if taxo else None,
        taxo.model_class if taxo else None,
        taxo.model_version if taxo else None,
        taxo.model_release if taxo else None,
    )


def _make_taxo_path(
    publisher: str | None, family: str | None, class_: str | None, version: str | None, release: str | None
) -> str:
    return f"/{publisher or ''}/{family or ''}/{class_ or ''}/{version or ''}/{release or ''}"


def _clean_keywords_list(raw_keywords: list[str | None]) -> list[str]:
    flattened = []
    for keyword in raw_keywords:
        if keyword is not None and keyword != "":
            flattened.extend(re.split(r"[ :()\[\]]+", keyword.lower()))
    # Filter out empty strings and return unique keywords
    return [keyword for keyword in set(flattened) if keyword]


def _create_keywords_from_taxonomy(taxo: LanguageModelTaxonomy, flags: list[str]) -> list[str]:
    keywords: list[str | None] = list(flags)
    if taxo.taxo_label:
        keywords.append(taxo.taxo_label)
    keywords.extend(
        [
            taxo.model_publisher,
            taxo.model_family,
            taxo.model_class,
            taxo.model_version,
        ]
    )
    return _clean_keywords_list(keywords)


async def get_model_taxonomies(query: ModelTaxonomyQuery) -> list[ModelTaxonomyResponse]:
    async with get_async_session() as session:
        sql_query = select(LanguageModelTaxonomy).join(LanguageModel)

        if query.pickable_only:
            sql_query = sql_query.where(LanguageModelTaxonomy.is_pickable.is_(True))  # type: ignore
        if query.leaf_node_only:
            sql_query = sql_query.where(LanguageModelTaxonomy.is_leaf_node.is_(True))  # type: ignore
        if query.has_active_providers:
            sql_query = sql_query.where(
                LanguageModelTaxonomy.language_model_taxonomy_id.in_(  # type: ignore
                    select(LanguageModelTaxonomy.language_model_taxonomy_id)
                    .join(LanguageModel)
                    .where(LanguageModel.status == LanguageModelStatusEnum.ACTIVE)
                    .distinct()
                    .scalar_subquery()
                )
            )

        if query.taxonomy_id:
            # search by taxonomy id, will ignore all other fields
            sql_query = sql_query.where(LanguageModelTaxonomy.language_model_taxonomy_id == query.taxonomy_id)
        elif any(
            x is not None
            for x in [
                query.model_publisher,
                query.model_family,
                query.model_class,
                query.model_version,
                query.model_release,
            ]
        ):
            if query.model_publisher is not None:
                if _is_none_or_null(query.model_publisher):
                    sql_query = sql_query.where(LanguageModelTaxonomy.model_publisher.is_(None))  # type: ignore
                else:
                    sql_query = sql_query.where(LanguageModelTaxonomy.model_publisher == query.model_publisher)

            if query.model_family is not None:
                if _is_none_or_null(query.model_family):
                    sql_query = sql_query.where(LanguageModelTaxonomy.model_family.is_(None))  # type: ignore
                else:
                    sql_query = sql_query.where(LanguageModelTaxonomy.model_family == query.model_family)

            if query.model_class is not None:
                if _is_none_or_null(query.model_class):
                    sql_query = sql_query.where(LanguageModelTaxonomy.model_class.is_(None))  # type: ignore
                else:
                    sql_query = sql_query.where(LanguageModelTaxonomy.model_class == query.model_class)

            if query.model_version is not None:
                if _is_none_or_null(query.model_version):
                    sql_query = sql_query.where(LanguageModelTaxonomy.model_version.is_(None))  # type: ignore
                else:
                    sql_query = sql_query.where(LanguageModelTaxonomy.model_version == query.model_version)

            if query.model_release is not None:
                if _is_none_or_null(query.model_release):
                    sql_query = sql_query.where(LanguageModelTaxonomy.model_release.is_(None))  # type: ignore
                else:
                    sql_query = sql_query.where(LanguageModelTaxonomy.model_release == query.model_release)
        else:
            # search with no condition, will return all taxonomy node entries.
            pass

        sql_query = sql_query.order_by(LanguageModelTaxonomy.priority.desc())  # type: ignore

        results = (await session.exec(sql_query)).all()

        response = []
        for result in results:
            flags = _infer_flags_from_taxonomy(result)
            keywords = _create_keywords_from_taxonomy(result, flags)
            response.append(
                ModelTaxonomyResponse(
                    **result.model_dump(
                        exclude={
                            "is_pro",
                            "is_strong",
                            "is_live",
                            "is_reasoning",
                            "is_image_generation",
                            "priority",
                        }
                    ),
                    taxonomy_id=result.language_model_taxonomy_id,
                    taxonomy_label=result.taxo_label,
                    taxonomy_path=_make_taxo_path_from_taxonomy(result),
                    flags=flags,
                    priority=result.priority or 0,
                    keywords=keywords,
                )
            )
        return response


def _create_keywords_from_model(model: LanguageModel, flags: list[str]) -> list[str]:
    keywords: list[str | None] = list(flags)
    if model.label:
        keywords.append(model.label)
    if model.taxonomy:
        if model.taxonomy.taxo_label:
            keywords.append(model.taxonomy.taxo_label)
        keywords.extend(
            [
                model.taxonomy.model_publisher,
                model.taxonomy.model_family,
                model.taxonomy.model_class,
                model.taxonomy.model_version,
            ]
        )
    return _clean_keywords_list(keywords)


def _is_new(created_at: datetime | None) -> bool:
    return created_at is not None and created_at > datetime.now(UTC) - timedelta(days=MAX_NEW_MODEL_AGE_DAYS)


def _infer_flags_from_model(model: LanguageModel) -> list[str]:
    flags = [
        flag
        for flag, value in {
            "NEW": _is_new(model.created_at),
            "PRO": model.is_pro,
            "STRONG": model.is_strong,
            "LIVE": model.is_live,
            "FAST": model.provider.is_fast,
            "REASONING": model.is_reasoning,
            "IMAGE_GENERATION": model.is_image_generation,
            "IMAGE": (model.supported_attachment_mime_types and supports_image(model.supported_attachment_mime_types)),
            "PDF": model.supported_attachment_mime_types and supports_pdf(model.supported_attachment_mime_types),
        }.items()
        if value
    ]
    return flags


def _infer_flags_from_taxonomy(taxo: LanguageModelTaxonomy) -> list[str]:
    # note that there is no "FAST" as it's a provider-related flag
    flags = [
        flag
        for flag, value in {
            "NEW": _is_new(taxo.created_at),
            "PRO": taxo.is_pro,
            "STRONG": taxo.is_strong,
            "LIVE": taxo.is_live,
            "REASONING": taxo.is_reasoning,
            "IMAGE_GENERATION": taxo.is_image_generation,
            "IMAGE": (taxo.supported_attachment_mime_types and supports_image(taxo.supported_attachment_mime_types)),
            "PDF": taxo.supported_attachment_mime_types and supports_pdf(taxo.supported_attachment_mime_types),
        }.items()
        if value
    ]
    return flags


def _create_language_model_struct(model: LanguageModel) -> LanguageModelStruct:
    flags = _infer_flags_from_model(model)
    keywords = _create_keywords_from_model(model, flags)

    return LanguageModelStruct(
        **model.model_dump(
            exclude={
                "organization",
                "model_publisher",
                "family",
                "model_class",
                "model_version",
                "model_release",
                "taxonomy_id",
                "priority",
            }
        ),
        model_publisher=model.taxonomy.model_publisher if model.taxonomy else model.model_publisher,
        model_family=model.taxonomy.model_family if model.taxonomy else model.family,
        model_class=model.taxonomy.model_class if model.taxonomy else model.model_class,
        model_version=model.taxonomy.model_version if model.taxonomy else model.model_version,
        model_release=model.taxonomy.model_release if model.taxonomy else model.model_release,
        organization_name=model.organization.organization_name if model.organization else None,
        provider_name=model.provider.name if model.provider else None,
        taxonomy_id=model.taxonomy_id,
        taxonomy_label=model.taxonomy.taxo_label if model.taxonomy else None,
        taxonomy_path=_make_taxo_path_from_taxonomy(model.taxonomy),
        flags=flags,
        keywords=keywords,
        priority=model.priority if model.priority is not None else 0,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def get_models(
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
    async with get_async_session() as session:
        query = select(LanguageModel).options(
            selectinload(LanguageModel.organization),  # type: ignore
            selectinload(LanguageModel.taxonomy),  # type: ignore
            selectinload(LanguageModel.provider),  # type: ignore
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

        models = (await session.exec(query)).all()
        # TODO(Tian): source other fields like parameter count/context window from taxonomy table as well.
        return [_create_language_model_struct(model) for model in models]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
# Only active and inactive models are returned as part of the model details as these
# have been onboarded and used for ranking in past
async def get_model_details(model_id: str) -> LanguageModelStruct | None:
    async with get_async_session() as session:
        query = (
            select(LanguageModel)
            .options(
                selectinload(LanguageModel.organization),  # type: ignore
                selectinload(LanguageModel.taxonomy),  # type: ignore
                selectinload(LanguageModel.provider),  # type: ignore
            )
            .where(
                LanguageModel.language_model_id == model_id,
                LanguageModel.deleted_at.is_(None),  # type: ignore
                LanguageModel.status.in_(  # type: ignore
                    [LanguageModelStatusEnum.ACTIVE, LanguageModelStatusEnum.INACTIVE]
                ),
            )
        )
        model = (await session.exec(query)).first()
        if model is None:
            return None
        return _create_language_model_struct(model)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def update_model(model_id: str, updated_model: LanguageModel) -> LanguageModelStruct:
    async with get_async_session() as session:
        model_data = updated_model.model_dump(exclude_unset=True, exclude={"language_model_id"})
        if not model_data:
            raise ValueError("No fields to update")

        existing_model = await session.get(LanguageModel, model_id)
        if existing_model is None:
            raise ValueError(f"Model with id {model_id} not found")

        for field, value in model_data.items():
            setattr(existing_model, field, value)

        existing_model.modified_at = func.now()  # type: ignore

        await session.commit()
        await session.refresh(existing_model)

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
async def delete_model(model_id: str) -> None:
    async with get_async_session() as session:
        result: Result = await session.exec(
            update(LanguageModel)
            .where(LanguageModel.language_model_id == model_id)  # type: ignore
            .values(deleted_at=func.now())
        )

        if result.rowcount == 0:  # type: ignore
            raise ValueError(f"Model with id {model_id} not found")
        await session.commit()


@async_timed_cache(seconds=600)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def get_model_base_statistics(model_id: str) -> LanguageModelStatistics:
    async with get_async_session() as session:
        llm = await session.get(LanguageModel, model_id)

        if llm is None:
            raise ValueError(f"Language model with id {model_id} not found")

        return LanguageModelStatistics(
            first_token_p50_latency_ms=llm.first_token_p50_latency_ms,
            first_token_p90_latency_ms=llm.first_token_p90_latency_ms,
            output_p50_tps=llm.output_p50_tps,
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
