import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel

if TYPE_CHECKING:
    from ypl.db.chats import SuggestedTrendingTopicPrompt


class TrendingTopic(BaseModel, table=True):
    """A trending topic from Google Trends.

    Columns are described in https://www.searchapi.io/docs/google-trends-trending-now-api.
    """

    __tablename__ = "trending_topics"

    trending_topic_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    position: int = Field(nullable=False)
    query: str = Field(nullable=False)
    location: str = Field(nullable=False)
    categories: list[str] = Field(sa_column=Column(ARRAY(String), nullable=False))
    start_date: datetime | None = Field(nullable=True, sa_type=DateTime(timezone=True))  # type: ignore
    end_date: datetime | None = Field(nullable=True, sa_type=DateTime(timezone=True))  # type: ignore
    is_active: bool | None = Field(nullable=True)
    keywords: list[str] = Field(sa_column=Column(ARRAY(String), nullable=False, server_default="{}"))

    news_stories: list["NewsStory"] = Relationship(
        back_populates="trending_topic", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    suggested_prompts: list["SuggestedTrendingTopicPrompt"] = Relationship(
        back_populates="trending_topic", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class NewsStory(BaseModel, table=True):
    """A news story related to a trending topic.

    Columns are described in https://www.searchapi.io/docs/google-trends-trending-now-news-api.
    """

    __tablename__ = "news_stories"

    news_story_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    title: str = Field(nullable=False)
    link: str = Field(nullable=False)
    source: str = Field(nullable=False)
    date: datetime | None = Field(nullable=True, sa_type=DateTime(timezone=True))  # type: ignore
    thumbnail: str | None = Field(nullable=True)

    trending_topic_id: uuid.UUID = Field(foreign_key="trending_topics.trending_topic_id", nullable=False, index=True)
    trending_topic: TrendingTopic = Relationship(back_populates="news_stories")
