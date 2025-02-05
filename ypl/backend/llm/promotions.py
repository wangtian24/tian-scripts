import logging
import math
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlmodel import Session
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.db import get_async_engine, get_async_session, get_engine
from ypl.backend.llm.db_helpers import get_model_creation_dates
from ypl.backend.llm.routing.modules.proposers import (
    ModelProposer,
)
from ypl.backend.llm.routing.policy import SelectionCriteria
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.utils.json import json_dumps
from ypl.db.language_models import LanguageModel
from ypl.db.model_promotions import ModelPromotion, ModelPromotionStatus
from ypl.utils import RNGMixin

MODEL_PROMO_DEFAULT_RANGE_HRS = 7 * 24  # hours in 7 days
MODEL_PROMO_HALF_LIFE_RATIO = 0.5  # the boost strength drops 50% at promo_days * half_life_ratio days

# The probability a promoted model will show up if it's proposed and it has full promo_strength (1.0).
# Technically you can set promo_strength > 1.0 to reach even higher show probability, but it's not recommended.
MODEL_PROMO_MAX_SHOW_PROB = 0.2


async def get_all_model_promotions() -> list[tuple[str, ModelPromotion]]:
    query = (
        select(LanguageModel.internal_name, ModelPromotion)  # type: ignore
        .join(ModelPromotion, ModelPromotion.language_model_id == LanguageModel.language_model_id)
        .where(ModelPromotion.deleted_at.is_(None))  # type: ignore
        .order_by(ModelPromotion.promo_start_date.desc())  # type: ignore
    )

    async with get_async_session() as session:
        results = await session.exec(query)
        if not results:
            return []
        return [(row[0], row[1]) for row in results]


def get_active_model_promotions() -> list[tuple[str, ModelPromotion]]:
    query = (
        select(LanguageModel.internal_name, ModelPromotion)  # type: ignore
        .join(ModelPromotion, ModelPromotion.language_model_id == LanguageModel.language_model_id)
        .where(
            ModelPromotion.promo_status == ModelPromotionStatus.ACTIVE,
            ModelPromotion.deleted_at.is_(None),  # type: ignore
            (ModelPromotion.promo_start_date.is_(None) | (ModelPromotion.promo_start_date <= func.current_timestamp())),  # type: ignore
            (ModelPromotion.promo_end_date.is_(None) | (ModelPromotion.promo_end_date >= func.current_timestamp())),  # type: ignore
        )
        .order_by(ModelPromotion.promo_start_date.desc())  # type: ignore
    )
    with Session(get_engine()) as session:
        results = session.exec(query).all()
        if not results:
            return []
        return [(row[0], row[1]) for row in results]


async def create_promotion(
    model_name: str, start_date: datetime | None = None, end_date: datetime | None = None, promo_strength: float = 1.0
) -> ModelPromotion:
    """Create a new promotion entry"""
    async with AsyncSession(get_async_engine()) as session:
        rows = (
            await session.exec(
                select(LanguageModel.language_model_id).where(LanguageModel.internal_name == model_name)  # type: ignore
            )
        ).first()
        if not rows or not rows[0]:
            raise ValueError(f"Model {model_name} not found")

        model_id = rows[0]
        promotion = ModelPromotion(
            language_model_id=model_id,
            promo_start_date=start_date,
            promo_end_date=end_date,
            promo_strength=promo_strength,
            promo_status=ModelPromotionStatus.INACTIVE,
        )

    async with AsyncSession(get_async_engine()) as session:
        session.add(promotion)
        await session.commit()
        await session.refresh(promotion)

    return promotion


async def activate_promotion(promotion_id: UUID) -> None:
    """Activate a promotion entry"""
    async with AsyncSession(get_async_engine()) as session:
        await session.exec(  # type: ignore
            update(ModelPromotion)
            .where(ModelPromotion.promotion_id == promotion_id)  # type: ignore
            .values(promo_status=ModelPromotionStatus.ACTIVE)
        )
        await session.commit()


async def deactivate_promotion(promotion_id: UUID) -> None:
    """Deactivate a promotion entry"""
    async with AsyncSession(get_async_engine()) as session:
        await session.exec(  # type: ignore
            update(ModelPromotion)
            .where(ModelPromotion.promotion_id == promotion_id)  # type: ignore
            .values(promo_status=ModelPromotionStatus.INACTIVE)
        )
        await session.commit()


class PromotionModelProposer(RNGMixin, ModelProposer):
    """
    Proposes models that are under active promotion and decide their show probability.
    """

    def _calc_damping(self, age_in_window: int, window_size: int) -> float:
        effective_age = min(age_in_window, window_size)  # the age cannot be larger than window size
        return float(1.0 / (1.0 + math.exp((effective_age - window_size * MODEL_PROMO_HALF_LIFE_RATIO) / 25.0)))

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if not models_to_select:
            return RouterState()

        # Get models in active promotion and their ages, in reverse time order of the starting time.
        # so if a model has multiple promotions, we will only respect the most recently started one.
        promoted_models: list[tuple[str, ModelPromotion]] = get_active_model_promotions()
        # if there's nothing to promote, or no promoted model is in the candidate set, skip
        if not promoted_models:
            return state
        if not any(model in models_to_select for model, _ in promoted_models):
            return state

        models_without_start_end = {
            model for model, promo in promoted_models if not promo.promo_start_date or not promo.promo_end_date
        }
        models_creation_dates: dict[str, datetime] = (
            get_model_creation_dates(tuple(models_without_start_end)) if models_without_start_end else {}
        )

        ld = {
            "message": f"Model Promotion: evaluating {len(promoted_models)} active promotions",
            "promotions": {
                model: {
                    "promo_start_date": promo.promo_start_date.isoformat() if promo.promo_start_date else None,
                    "promo_end_date": promo.promo_end_date.isoformat() if promo.promo_end_date else None,
                    "promo_strength": promo.promo_strength,
                }
                for model, promo in promoted_models
            },
        }
        logging.info(json_dumps(ld))

        # Calculate the age-in-window dependent dampings for all eligible models
        dampings: dict[str, float] = {}  # time-window dependent damping factor
        strengths: dict[str, float] = {}  # promotion strength
        current_time = datetime.now(UTC)
        for model, promotion in promoted_models:
            if model in dampings:
                continue

            # get the age of the model
            if promotion.promo_start_date and promotion.promo_end_date:
                # start/end date both exist, will use this range.
                age_in_window = int((current_time - promotion.promo_start_date).total_seconds() / 3600)
                window_size = int((promotion.promo_end_date - promotion.promo_start_date).total_seconds() / 3600)
            else:
                # promotion without start and/or end date, the default is (creation_date, creation_date + 7 days)
                age_in_window = int((current_time - models_creation_dates[model]).total_seconds() / 3600)
                window_size = MODEL_PROMO_DEFAULT_RANGE_HRS

            dampings[model] = self._calc_damping(age_in_window, window_size)
            strengths[model] = promotion.promo_strength or 1.0

        # Sample from all promotion candidate models based on their time window damping.
        candidates_names = list(dampings.keys())
        candidates_weights = list(dampings.values())
        candidates_weights_sum = sum(candidates_weights)
        candidates_weights_normalized = [w / candidates_weights_sum for w in candidates_weights]  # normalize
        chosen_model = str(self.get_rng().choice(candidates_names, p=candidates_weights_normalized))

        # The final show probability depends on where a promotion is in its time window and its strength.
        show_probability = min(1.0, MODEL_PROMO_MAX_SHOW_PROB * dampings[chosen_model] * strengths[chosen_model])
        if self.get_rng().random() > show_probability:
            return state

        log_dict = {
            "message": f"Model Promotion: picked {chosen_model} with probability = {show_probability:.3f}",
            "promo_id": promotion.promotion_id,
            "promo_start_date": promotion.promo_start_date.isoformat() if promotion.promo_start_date else None,
            "promo_end_date": promotion.promo_end_date.isoformat() if promotion.promo_end_date else None,
            "promo_strength": strengths[chosen_model],
            "age_in_window_hrs": age_in_window,
            "window_size_hrs": window_size,
            "damping": dampings[chosen_model],
        }
        logging.info(json_dumps(log_dict))

        return state.emplaced(
            selected_models={chosen_model: {SelectionCriteria.PROMOTED_MODELS: show_probability}},
            all_models=state.all_models,
            excluded_models=state.excluded_models | (state.all_models - set(models_to_select)),
        )

    async def _apropose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        return self._propose_models(models_to_select, state)
