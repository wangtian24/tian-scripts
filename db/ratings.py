import uuid

from sqlalchemy import Float, ForeignKey, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import BaseModel


class Category(BaseModel):
    __tablename__ = "categories"

    category_id = mapped_column(Uuid(as_uuid=True), primary_key=True, nullable=False, default=uuid.uuid4)

    name = mapped_column(String, nullable=False)
    description = mapped_column(String, nullable=True)

    # The top level category has no parent.
    parent_category_id = mapped_column(Uuid(as_uuid=True), ForeignKey("categories.category_id"), nullable=True)
    parent_category = relationship("Category", remote_side="Category.category_id", back_populates="child_categories")
    child_categories: Mapped[list["Category"]] = relationship("Category", back_populates="parent_category")

    def __str__(self) -> str:
        return self.get_hierarchical_name()

    def get_hierarchical_name(self) -> str:
        name_parts = [self.name]
        parent = self.parent_category
        while parent is not None:
            name_parts.append(parent.name)
            parent = parent.parent_category
        return " > ".join(reversed(name_parts))


# A rating is a score for a model in a category.
class Rating(BaseModel):
    __tablename__ = "ratings"

    rating_id = mapped_column(Uuid(as_uuid=True), primary_key=True, nullable=False, default=uuid.uuid4)
    model_id = mapped_column(Uuid(as_uuid=True), ForeignKey("language_models.model_id"), nullable=False)
    category_id = mapped_column(Uuid(as_uuid=True), ForeignKey("categories.category_id"), nullable=False)
    score = mapped_column(Float, nullable=False, default=0)

    # The lower and upper bounds of the 95% confidence interval for the score.
    lower_bound_95 = mapped_column(Float, nullable=False, default=0)
    upper_bound_95 = mapped_column(Float, nullable=False, default=0)

    model = relationship("LanguageModel", back_populates="ratings")
    category = relationship("Category", back_populates="ratings")

    # A model can have different ratings for different categories.
    __table_args__ = (UniqueConstraint("model_id", "category_id", name="uq_model_category"),)
