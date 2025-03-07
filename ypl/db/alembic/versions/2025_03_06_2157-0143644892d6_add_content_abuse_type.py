"""add content abuse type

Revision ID: 0143644892d6
Revises: bece937e9274
Create Date: 2025-03-06 21:57:57.565801+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic_postgresql_enum import TableReference

# revision identifiers, used by Alembic.
revision: str = '0143644892d6'
down_revision: str | None = 'bece937e9274'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'abuseeventtype', ['CASHOUT_SAME_INSTRUMENT_AS_REFERRER', 'CASHOUT_SAME_INSTRUMENT_AS_RECENT_NEW_USER', 'CASHOUT_MULTIPLE_RECENT_REFERRAL_SIGNUPS', 'SIGNUP_SAME_EMAIL_AS_EXISTING_USER', 'SIGNUP_SIMILAR_EMAIL_AS_REFERRER', 'SIGNUP_SIMILAR_EMAIL_AS_RECENT_USER', 'SIGNUP_SIMILAR_NAME_AS_REFERRER', 'SIGNUP_SIMILAR_NAME_AS_RECENT_USER', 'ACTIVITY_VOLUME', 'CONTENT_LOW_QUALITY_MODEL_FEEDBACK'],
                        [TableReference(table_schema='public', table_name='abuse_events', column_name='event_type')],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'abuseeventtype', ['CASHOUT_SAME_INSTRUMENT_AS_REFERRER', 'CASHOUT_SAME_INSTRUMENT_AS_RECENT_NEW_USER', 'CASHOUT_MULTIPLE_RECENT_REFERRAL_SIGNUPS', 'SIGNUP_SAME_EMAIL_AS_EXISTING_USER', 'SIGNUP_SIMILAR_EMAIL_AS_REFERRER', 'SIGNUP_SIMILAR_EMAIL_AS_RECENT_USER', 'SIGNUP_SIMILAR_NAME_AS_REFERRER', 'SIGNUP_SIMILAR_NAME_AS_RECENT_USER', 'ACTIVITY_VOLUME'],
                        [TableReference(table_schema='public', table_name='abuse_events', column_name='event_type')],
                        enum_values_to_rename=[])
    # ### end Alembic commands ###
