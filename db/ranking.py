import uuid

from sqlalchemy import ForeignKey, String, Uuid
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
