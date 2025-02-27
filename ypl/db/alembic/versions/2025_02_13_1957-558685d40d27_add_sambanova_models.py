"""add sambanova models

Revision ID: 558685d40d27
Revises: f650d72412a0
Create Date: 2025-02-13 19:57:30.230941+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlmodel import Session, select

from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Organization, Provider


# revision identifiers, used by Alembic.
revision: str = '558685d40d27'
down_revision: str | None = 'f650d72412a0'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
SAMBANOVA_MODELS = [
    {
        "label": "Deepseek-R1 Distill Llama 70B (Sambanova)",
        "family": "Deepseek-R1",
        "internal_name": "DeepSeek-R1-Distill-Llama-70B",
        "parameter_count": 70_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": True,
        "semantic_group": "deepseek r1",
    },
    {
        "label": "Llama 3.1 Tulu 3 405B (Sambanova)",
        "family": "Llama 3.1",
        "internal_name": "Llama-3.1-Tulu-3-405B",
        "parameter_count": 405_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": True,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.1 405B Instruct (Sambanova)",
        "family": "Llama 3.1",
        "internal_name": "Meta-Llama-3.1-405B-Instruct",
        "parameter_count": 405_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": True,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.1 70B Instruct (Sambanova)",
        "family": "Llama 3.1",
        "internal_name": "Meta-Llama-3.1-70B-Instruct",
        "parameter_count": 70_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": True,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.1 8B Instruct (Sambanova)",
        "family": "Llama 3.1",
        "internal_name": "Meta-Llama-3.1-8B-Instruct",
        "parameter_count": 8_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": False,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.2 1B Instruct (Sambanova)",
        "family": "Llama 3.2",
        "internal_name": "Meta-Llama-3.2-1B-Instruct",
        "parameter_count": 1_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": False,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.2 3B Instruct (Sambanova)",
        "family": "Llama 3.2",
        "internal_name": "Meta-Llama-3.2-3B-Instruct",
        "parameter_count": 3_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": False,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.3 70B Instruct (Sambanova)",
        "family": "Llama 3.3",
        "internal_name": "Meta-Llama-3.3-70B-Instruct",
        "parameter_count": 70_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": True,
        "semantic_group": "llama",
    },
    {
        "label": "Qwen2.5 72B Instruct (Sambanova)",
        "family": "Qwen2.5 72B",
        "internal_name": "Qwen2.5-72B-Instruct",
        "parameter_count": 72_000_000_000,
        "context_window_tokens": 131_072,
        "is_strong": True,
        "semantic_group": "qwen",
    },
    {
        "label": "Qwen2.5 Coder 32B Instruct (Sambanova)",
        "family": "Qwen2.5 Coder",
        "internal_name": "Qwen2.5-Coder-32B-Instruct",
        "parameter_count": 32_000_000_000,
        "context_window_tokens": 131_072,
        "is_strong": False,
        "semantic_group": "qwen",
    },
    {
        "label": "QwQ 32B Preview (Sambanova)",
        "family": "QwQ 32B",
        "internal_name": "QwQ-32B-Preview",
        "parameter_count": 32_000_000_000,
        "context_window_tokens": 32_768,
        "is_strong": False,
        "semantic_group": "qwen",
    },
    {
        "label": "Llama 3.2 11B Vision Instruct (Sambanova)",
        "family": "Llama 3.2",
        "internal_name": "Llama-3.2-11B-Vision-Instruct",
        "parameter_count": 11_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": False,
        "supported_attachement_mime_types": ["image/*"],
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.2 90B Vision Instruct (Sambanova)",
        "family": "Llama 3.2",
        "internal_name": "Llama-3.2-90B-Vision-Instruct",
        "parameter_count": 90_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": True,
        "supported_attachement_mime_types": ["image/*"],
        "semantic_group": "llama",
    },
]

AVARTAR_URL = "https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/sambanova%20logo-DQufRZ9YknZF3bE2WoH6y0LQ2zlyuJ.webp"


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with Session(op.get_bind()) as session:
        # add organization
        org_id = session.exec(select(Organization.organization_id).where(Organization.organization_name == "Sambanova")).first()
        if not org_id:
            query = sa.text("""
                INSERT INTO organizations (
                    organization_id, organization_name, created_at
                ) VALUES (
                    GEN_RANDOM_UUID(), :organization_name, NOW()
                ) RETURNING organization_id""")
            org_id = session.execute(query, {"organization_name": "Sambanova"}).fetchone()[0]
            session.commit()

        # add provider
        provider_id = session.exec(select(Provider.provider_id).where(Provider.name == "Sambanova")).first()
        if not provider_id:
            query = sa.text("""
                INSERT INTO providers (
                    provider_id, name, base_api_url, api_key_env_name, is_active, created_at
                ) VALUES (
                    GEN_RANDOM_UUID(), :name, :base_api_url, :api_key_env_name, :is_active, NOW()
                ) RETURNING provider_id
            """)
            provider_id = session.execute(query, {
                "name": "Sambanova",
                "base_api_url": "https://api.sambanova.ai/v1",
                "api_key_env_name": "SAMBANOVA_API_KEY",
                "is_active": True,
            }).fetchone()[0]
            session.commit()

        # add all models
        for model in SAMBANOVA_MODELS:
            existing_model_id = session.exec(select(LanguageModel.language_model_id).where(
                LanguageModel.internal_name == model["internal_name"]
            )).first()
            
            if not existing_model_id:
                session.execute(
                    sa.text("""
                        INSERT INTO language_models (
                            created_at, language_model_id,
                            provider_id, organization_id, name, creator_user_id, 
                            avatar_url, is_pro, label, family, internal_name,
                            parameter_count, context_window_tokens, is_strong,
                            semantic_group
                        ) VALUES (
                            NOW(), GEN_RANDOM_UUID(),
                            :provider_id, :org_id, :name, :creator_user_id,
                            :avatar_url, :is_pro, :label, :family, :internal_name,
                            :parameter_count, :context_window_tokens, :is_strong,
                            :semantic_group
                        )
                    """),
                    {
                        "provider_id": provider_id,
                        "org_id": org_id,
                        "name": "sambanova/" + model["internal_name"],
                        "creator_user_id": "SYSTEM",
                        "avatar_url": AVARTAR_URL,
                        "is_pro": False,
                        "label": model["label"],
                        "family": model["family"],
                        "internal_name": model["internal_name"],
                        "parameter_count": model["parameter_count"],
                        "context_window_tokens": model["context_window_tokens"],
                        "is_strong": model["is_strong"],
                        "status": "SUBMITTED",
                        "semantic_group": model["semantic_group"]
                    }
                )
        session.commit()

    # ### end Alembic commands ###


def downgrade() -> None:
    # # ### commands auto generated by Alembic - please adjust! ###
    pass # don't really do anything, no need to delete
    # ### end Alembic commands ###
