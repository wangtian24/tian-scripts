import uuid

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship

from db.base import BaseModel
from db.language_models import LanguageModel


class Category(BaseModel, table=True):
    __tablename__ = "categories"

    category_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(nullable=False)
    description: str | None = Field(default=None)

    parent_category_id: uuid.UUID | None = Field(default=None, foreign_key="categories.category_id")
    parent_category: "Category" = Relationship(
        back_populates="child_categories", sa_relationship_kwargs={"remote_side": "Category.category_id"}
    )
    child_categories: list["Category"] = Relationship(back_populates="parent_category")

    def __str__(self) -> str:
        return self.get_hierarchical_name()

    def get_hierarchical_name(self) -> str:
        name_parts = [self.name]
        parent = self.parent_category
        while parent is not None:
            name_parts.append(parent.name)
            parent = parent.parent_category
        return " > ".join(reversed(name_parts))


class Rating(BaseModel, table=True):
    __tablename__ = "ratings"
    __table_args__ = (UniqueConstraint("model_id", "category_id", name="uq_model_category"),)

    rating_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    model_id: uuid.UUID = Field(foreign_key="language_models.model_id")
    category_id: uuid.UUID = Field(foreign_key="categories.category_id")
    score: float = Field(default=0)
    lower_bound_95: float = Field(default=0)
    upper_bound_95: float = Field(default=0)

    model: LanguageModel = Relationship(back_populates="ratings")
    category: Category = Relationship(back_populates="ratings")
