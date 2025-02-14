"""add routing rules and yapp details

Revision ID: 3401935f99f2
Revises: 3e6ce97ce0d2
Create Date: 2025-02-12 20:08:54.095940+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlmodel import Session, select

from ypl.db.language_models import LanguageModel, RoutingAction, RoutingRule
from ypl.db.yapps import Yapp


# revision identifiers, used by Alembic.
revision: str = '3401935f99f2'
down_revision: str | None = '3e6ce97ce0d2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SOURCE_CATEGORIES = [
    # category (same as yapp model name), z-index, destination, probability
    ("weather-yapp", 1_000_000, "Yapp Temporary/weather-yapp", 1.0),
    ("news-yapp", 1_000_000, "Yapp Temporary/news-yapp", 0.4),
    ("wikipedia-yapp", 1_000_000, "Yapp Temporary/wikipedia-yapp", 0.4),
    (
        "youtube-transcript-yapp",
        1_000_000,
        "Yapp Temporary/youtube-transcript-yapp",
        1.0,
    ),
]

YAPPS_DESC = [
    (
        "weather-yapp",
        "gathers and summarizes real-time weather information and meteorological data to deliver accurate forecasts and insights",
    ),
    (
        "news-yapp",
        "aggregates, filters, and synthesizes current news articles from diverse sources to deliver concise, real-time insights on ongoing events.",
    ),
    (
        "wikipedia-yapp",
        "searches, retrieves, and synthesizes Wikipedia content to help users find information on a wide range of time-insensitive topics.",
    ),
    (
        "youtube-transcript-yapp",
        "retrieves YouTube video transcripts, summarizes key points, provide quick insights, and answers questions about videos.",
    ),
]


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with Session(op.get_bind()) as session:
        # Add data to just-created yapps table
        for yapp_model_name, yapp_desc in YAPPS_DESC:
            lang_id = session.exec(select(LanguageModel.language_model_id).where(LanguageModel.internal_name == yapp_model_name)).first()
            if lang_id:
                yapp = Yapp(
                    language_model_id=lang_id,
                    description=yapp_desc,
                )
                session.add(yapp)
            else:
                print(f"Warning: Language model {yapp_model_name} not found, Yapp entry not added")


        # Add data to routing table
        for category, z_index, destination, probability in SOURCE_CATEGORIES:
            routing_rule = RoutingRule(
                source_category=category,
                is_active=True,
                destination=destination,
                z_index=z_index,
                target=RoutingAction.ACCEPT,
                probability=probability,
            )
            session.add(routing_rule)

        session.commit()
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Delete routing rules
    with Session(op.get_bind()) as session:
        session.exec(
            sa.delete(RoutingRule).where(
                RoutingRule.source_category.in_(
                    [category for category, _, _, _ in SOURCE_CATEGORIES]
                )
            )
        )
        session.commit()
    # ### end Alembic commands ###
