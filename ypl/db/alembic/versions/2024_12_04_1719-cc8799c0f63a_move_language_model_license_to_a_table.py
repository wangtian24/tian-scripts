"""move language model license to a table

Revision ID: cc8799c0f63a
Revises: 860508bf689b
Create Date: 2024-12-04 17:19:51.264713+00:00

"""
from collections.abc import Sequence
import re
from uuid import uuid4

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.orm import Session

from ypl.db.language_models import LanguageModel, LanguageModelLicense, LicenseEnum


# revision identifiers, used by Alembic.
revision: str = 'cc8799c0f63a'
down_revision: str | None = '860508bf689b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('language_model_licenses',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('modified_at', sa.DateTime(timezone=True), server_default=sa.text("(now() AT TIME ZONE 'utc')"), nullable=True),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('language_model_license_id', sa.Uuid(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.PrimaryKeyConstraint('language_model_license_id', name=op.f('pk_language_model_licenses'))
    )
    op.create_index(op.f('ix_language_model_licenses_name'), 'language_model_licenses', ['name'], unique=True)
    op.add_column('language_models', sa.Column('language_model_license_id', sa.Uuid(), nullable=True))
    op.create_foreign_key(op.f('fk_language_models_language_model_license_id_language_model_licenses'), 'language_models', 'language_model_licenses', ['language_model_license_id'], ['language_model_license_id'])

    session = Session(bind=op.get_bind())

    # Populate license table using existing enum values.
    licenses = {}
    for license_enum in LicenseEnum:
        # The current name uses underscores instead of hyphens and periods.
        # Replace underscores after numbers with periods (these are version numbers).
        # Replace other underscores with hyphens.
        name = re.sub(r'(\d)_', r'\1.', license_enum.name).replace("_", "-")
        license = LanguageModelLicense(language_model_license_id=uuid4(), name=name)
        session.add(license)
        licenses[license_enum.name] = license
    
    session.flush()
    
    # Update language models with the new license.
    default_license = licenses[LicenseEnum.unknown.name]
    language_models = session.query(LanguageModel).all()
    for model in language_models:
        model.language_model_license = licenses.get(model.license.name, default_license)    

    session.flush()
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(op.f('fk_language_models_language_model_license_id_language_model_licenses'), 'language_models', type_='foreignkey')
    op.drop_column('language_models', 'language_model_license_id')
    op.drop_index(op.f('ix_language_model_licenses_name'), table_name='language_model_licenses')
    op.drop_table('language_model_licenses')
    # ### end Alembic commands ###
