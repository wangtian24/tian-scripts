import logging
import math
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, text, update
from sqlmodel import Session

from ypl.backend.db import get_engine
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

MODEL_PROMO_DEFAULT_RANGE_HRS = 7 * 24  # 7 days
MODEL_PROMO_HALF_LIFE_RATIO = 0.5  # the boost strength drops 50% at promo_days * half_life_ratio days

# The probability a promoted model will show up if it's proposed and it has full promo_strength (1.0).
# Technically you can set promo_strength > 1.0 to reach even higher show probability, but it's not recommended.
MODEL_PROMO_MAX_SHOW_PROB = 0.2


def get_active_model_promotions() -> list[tuple[str, ModelPromotion]]:
    sql_query = text(
        """
        SELECT lm.internal_name, mp.promo_status, mp.promo_start_date, mp.promo_end_date, mp.promo_strength
        FROM model_promotions mp
            JOIN language_models lm ON mp.language_model_id = lm.language_model_id
        WHERE mp.promo_status = 'ACTIVE'
            AND mp.deleted_at IS NULL
            AND (mp.promo_start_date IS NULL OR mp.promo_start_date <= CURRENT_TIMESTAMP)
            AND (mp.promo_end_date IS NULL OR mp.promo_end_date >= CURRENT_TIMESTAMP)
        ORDER BY mp.promo_start_date DESC
    """
    )
    with Session(get_engine()) as session:
        results = session.exec(sql_query).all()  # type: ignore
        if not results:
            return []
        return [
            (
                row.internal_name,
                ModelPromotion(
                    promo_status=row.promo_status,
                    promo_start_date=row.promo_start_date,
                    promo_end_date=row.promo_end_date,
                    promo_strength=row.promo_strength,
                ),
            )
            for row in results
        ]


async def create_promotion(
    model_name: str, start_date: datetime | None = None, end_date: datetime | None = None, promo_strength: float = 1.0
) -> ModelPromotion:
    """Create a new promotion entry"""
    with Session(get_engine()) as session:
        rows = session.exec(
            select(LanguageModel.language_model_id).where(LanguageModel.internal_name == model_name)  # type: ignore
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

    with Session(get_engine()) as session:
        session.add(promotion)
        session.commit()
        session.refresh(promotion)

    return promotion


def activate_promotion(promotion_id: UUID) -> None:
    """Activate a promotion entry"""
    with Session(get_engine()) as session:
        session.exec(  # type: ignore
            update(ModelPromotion)
            .where(ModelPromotion.promotion_id == promotion_id)  # type: ignore
            .values(promo_status=ModelPromotionStatus.ACTIVE)
        )
        session.commit()


def deactivate_promotion(promotion_id: UUID) -> None:
    """Deactivate a promotion entry"""
    with Session(get_engine()) as session:
        session.exec(  # type: ignore
            update(ModelPromotion)
            .where(ModelPromotion.promotion_id == promotion_id)  # type: ignore
            .values(promo_status=ModelPromotionStatus.INACTIVE)
        )
        session.commit()


class PromotionModelProposer(RNGMixin, ModelProposer):
    """
    Proposes models that are under active promotion and decide their show probability.
    """

    def _calc_damping(self, age: int, promo_range: int) -> float:
        return float(1.0 / (1.0 + math.exp((age - promo_range * MODEL_PROMO_HALF_LIFE_RATIO) / 2.0)))

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
            "message": f"Picked model for promotion: {chosen_model} with probability = {show_probability:.3f}",
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
