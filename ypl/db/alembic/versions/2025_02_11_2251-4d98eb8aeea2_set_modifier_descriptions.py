"""set modifier descriptions

Revision ID: f27b2d83b618
Revises: cc84d8417cef
Create Date: 2025-02-12 17:05:44.726547+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from collections.abc import Sequence
from sqlmodel import Session
from ypl.db.chats import PromptModifier

# revision identifiers, used by Alembic.
revision: str = 'f27b2d83b618'
down_revision: str | None = 'cc84d8417cef'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


DESCRIPTIONS = {
    "Shorter": "Makes responses more concise and direct.",
    "More structured": "Organizes responses for clarity and readability.",
    "More formal": "Enhances professionalism with clear, formal language.",
    "More casual": "Makes responses friendly and conversational.",
    "Tabular": "Formats responses into clear, structured tables wherever applicable.",
    "Summarize": "Provides a clear, structured summary.",
}

def upgrade() -> None:
    with Session(op.get_bind()) as session:
        for modifier_name, description in DESCRIPTIONS.items():
            session.exec(
                sa.update(PromptModifier)
                .where(PromptModifier.name == modifier_name)
                .values(description=description)
            )
        session.commit()

def downgrade() -> None:
    pass
