import uuid
from datetime import datetime
from enum import Enum

import sqlalchemy as sa
from sqlalchemy import Column
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel
from ypl.db.language_models import LanguageModel


class ModelPromotionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class ModelPromotion(BaseModel, table=True):
    """
    Stores promotion information for models. We can run promotion campaigns so they got boosts in routing and
    gain more exposure.
    """

    __tablename__ = "model_promotions"

    promotion_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)

    # The language model used to generate the message. Set when the message_type is assistant_message.
    language_model_id: uuid.UUID | None = Field(foreign_key="language_models.language_model_id", nullable=False)

    promo_status: ModelPromotionStatus = Field(default=ModelPromotionStatus.INACTIVE, nullable=False)

    # When the promotion starts
    promo_start_date: datetime | None = Field(sa_column=Column(sa.DateTime(timezone=True), nullable=True, default=None))
    # When the promotion ends
    promo_end_date: datetime | None = Field(sa_column=Column(sa.DateTime(timezone=True), nullable=True, default=None))

    # Strength of selection: controls how likely this model will be proposed over other promoted models, > 0
    # if model A has value 0.5 and model B has value 1.0, they will be proposed at probability of 1/3 vs 2/3 if they are
    # both at the beginning of their promotion window.
    # The value should be > 0
    proposal_strength: float | None = Field(sa_column=Column(sa.Float, nullable=True, default=1.0))

    # Promotion strength: controls how likely this model will actually show up once proposed
    # if model A has value 0.3, after it has won the proposal, it will have a X*0.3 chance to actually be injected.
    # The value of X is defined in MODEL_PROMO_MAX_SHOW_PROB in promotions.py.
    # The value should be > 0
    promo_strength: float | None = Field(sa_column=Column(sa.Float, nullable=True, default=1.0))

    __table_args__ = (
        sa.CheckConstraint(
            "promo_strength > 0.0",
            name="model_promotions_promo_strength_range",
        ),
        sa.CheckConstraint(
            "proposal_strength > 0.0",
            name="model_promotions_proposal_strength_range",
        ),
        sa.CheckConstraint(
            "(promo_start_date IS NULL OR promo_end_date IS NULL) OR promo_end_date > promo_start_date",
            name="model_promotions_date_order",
        ),
    )

    # --- Relationships ---
    language_model: "LanguageModel" = Relationship(back_populates="promotions")
