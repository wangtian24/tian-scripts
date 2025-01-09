"""add modifier UI status

Revision ID: be70aa19ea46
Revises: 639895bbd129
Create Date: 2025-01-09 02:21:52.696237+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic_postgresql_enum import TableReference
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'be70aa19ea46'
down_revision: str | None = '639895bbd129'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    sa.Enum('SELECTED', 'HIDDEN', name='messagemodifierstatus').create(op.get_bind())
    op.add_column('chat_messages', sa.Column('modifier_status', postgresql.ENUM('SELECTED', 'HIDDEN', name='messagemodifierstatus', create_type=False), nullable=True))
    op.sync_enum_values('public', 'messageuistatus', ['UNKNOWN', 'SEEN', 'DISMISSED', 'SELECTED'],
                        [TableReference(table_schema='public', table_name='chat_messages', column_name='ui_status', existing_server_default="'UNKNOWN'::messageuistatus")],
                        enum_values_to_rename=[])

    # If there are turns with multiple messages from the same assistant, set a random message as SELECTED.
    op.execute("""
        WITH message_groups AS (
            SELECT
                turn_id,
                assistant_language_model_id,
                array_agg(message_id) as message_ids
            FROM chat_messages
            WHERE assistant_language_model_id IS NOT null
            GROUP BY turn_id, assistant_language_model_id
            HAVING COUNT(*) > 1
        ),
        selected_messages AS (
            SELECT
                m.message_id,
                mg.turn_id,
                mg.assistant_language_model_id,
                CASE
                    WHEN m.message_id = mg.message_ids[1]
                    THEN 'SELECTED'::messagemodifierstatus
                    ELSE 'HIDDEN'::messagemodifierstatus
                END as new_status
            FROM message_groups mg
            CROSS JOIN LATERAL unnest(mg.message_ids) as m(message_id)
        )
        UPDATE chat_messages
        SET modifier_status = selected_messages.new_status
        FROM selected_messages
        WHERE chat_messages.message_id = selected_messages.message_id;
    """)

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values('public', 'messageuistatus', ['UNKNOWN', 'SEEN', 'DISMISSED', 'SELECTED', 'HIDDEN'],
                        [TableReference(table_schema='public', table_name='chat_messages', column_name='ui_status', existing_server_default="'UNKNOWN'::messageuistatus")],
                        enum_values_to_rename=[])
    op.drop_column('chat_messages', 'modifier_status')
    sa.Enum('SELECTED', 'HIDDEN', name='messagemodifierstatus').drop(op.get_bind())
    # ### end Alembic commands ###
