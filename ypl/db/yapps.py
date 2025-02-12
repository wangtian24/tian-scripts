import uuid

from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel
from ypl.db.language_models import LanguageModel


class Yapp(BaseModel, table=True):
    __tablename__ = "yapps"

    yapp_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    language_model_id: uuid.UUID = Field(foreign_key="language_models.language_model_id", nullable=False, index=True)
    description: str = Field(nullable=False)

    language_model: LanguageModel = Relationship(back_populates="yapps")
