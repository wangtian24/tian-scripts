import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship

from db.base import BaseModel
from db.language_models import LanguageModel

# The "category" used for the overall ranking (the ranking across all categories).
OVERALL_CATEGORY_NAME = "Overall"
OVERALL_CATEGORY_DESCRIPTION = "Overall ranking across all categories"


class Category(BaseModel, table=True):
    __tablename__ = "categories"
    __table_args__ = (UniqueConstraint("name", "parent_category_id", name="uq_name_parent"),)

    category_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(nullable=False)
    description: str | None = Field(default=None)

    parent_category_id: uuid.UUID | None = Field(default=None, foreign_key="categories.category_id")
    parent_category: "Category" = Relationship(
        back_populates="child_categories", sa_relationship_kwargs={"remote_side": "Category.category_id"}
    )
    child_categories: list["Category"] = Relationship(back_populates="parent_category")
    ratings: list["Rating"] = Relationship(back_populates="category")
    ratings_history: list["RatingHistory"] = Relationship(back_populates="category")

    def __str__(self) -> str:
        return self.get_hierarchical_name()

    def get_hierarchical_name(self) -> str:
        name_parts = [self.name]
        parent = self.parent_category
        while parent is not None:
            name_parts.append(parent.name)
            parent = parent.parent_category
        return " > ".join(reversed(name_parts))


# Ratings for a model in a category at a given created_at timestamp. When eval
# batch processing is run, new rows are created in this table for each
# model-category pair.
class RatingHistory(BaseModel, table=True):
    __tablename__ = "ratings_history"
    __table_args__ = (
        UniqueConstraint("language_model_id", "category_id", "created_at", name="uq_model_category_created_at"),
    )

    rating_history_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    language_model_id: uuid.UUID = Field(foreign_key="language_models.language_model_id")
    category_id: uuid.UUID = Field(foreign_key="categories.category_id")
    score: float = Field(default=0)
    score_lower_bound_95: float = Field(default=0)
    score_upper_bound_95: float = Field(default=0)

    model: LanguageModel = Relationship(back_populates="ratings_history")
    category: Category = Relationship(back_populates="ratings_history")

    snapshot_timestamp: datetime = Field(sa_type=sa.DateTime(timezone=True))  # type: ignore

    wins: int = Field(default=0)
    losses: int = Field(default=0)
    ties: int = Field(default=0)

    ratings: list["Rating"] = Relationship(back_populates="history")


# The latest rating for a model in a category. This serves a pointer to a row
# in the ratings_history table for fast access of the latest rating. When eval
# batch processing is run, rating_history_id is set to the latest
# rating_history_id.
class Rating(BaseModel, table=True):
    __tablename__ = "ratings"
    __table_args__ = (UniqueConstraint("language_model_id", "category_id", name="uq_model_category"),)

    rating_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    language_model_id: uuid.UUID = Field(foreign_key="language_models.language_model_id")
    category_id: uuid.UUID = Field(foreign_key="categories.category_id")

    # Points to the latest row in the ratings_history table.
    rating_history_id: uuid.UUID = Field(foreign_key="ratings_history.rating_history_id")

    model: LanguageModel = Relationship(back_populates="ratings")
    category: Category = Relationship(back_populates="ratings")

    history: list[RatingHistory] = Relationship(back_populates="ratings")
