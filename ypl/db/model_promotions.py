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
    # Strength of promotion (higher = stronger promotion, must be positive. 1.0 mean regular exposure)
    promo_strength: float | None = Field(sa_column=Column(sa.Float, nullable=True, default=None))
    __table_args__ = (
        sa.CheckConstraint(
            "promo_strength > 0.0",
            name="model_promotions_promo_strength_range",
        ),
        sa.CheckConstraint(
            "(promo_start_date IS NULL OR promo_end_date IS NULL) OR promo_end_date > promo_start_date",
            name="model_promotions_date_order",
        ),
    )

    # --- Relationships ---
    language_model: "LanguageModel" = Relationship(back_populates="promotions")
