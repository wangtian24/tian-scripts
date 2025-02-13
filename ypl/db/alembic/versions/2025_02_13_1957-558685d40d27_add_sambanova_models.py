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
        "status": LanguageModelStatusEnum.SUBMITTED,
        "semantic_group": "deepseek r1",
    },
    {
        "label": "Llama 3.1 Tulu 3 405B (Sambanova)",
        "family": "Llama 3.1",
        "internal_name": "Llama-3.1-Tulu-3-405B",
        "parameter_count": 405_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": True,
        "status": LanguageModelStatusEnum.SUBMITTED,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.1 405B Instruct (Sambanova)",
        "family": "Llama 3.1",
        "internal_name": "Meta-Llama-3.1-405B-Instruct",
        "parameter_count": 405_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": True,
        "status": LanguageModelStatusEnum.SUBMITTED,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.1 70B Instruct (Sambanova)",
        "family": "Llama 3.1",
        "internal_name": "Meta-Llama-3.1-70B-Instruct",
        "parameter_count": 70_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": True,
        "status": LanguageModelStatusEnum.SUBMITTED,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.1 8B Instruct (Sambanova)",
        "family": "Llama 3.1",
        "internal_name": "Meta-Llama-3.1-8B-Instruct",
        "parameter_count": 8_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": False,
        "status": LanguageModelStatusEnum.SUBMITTED,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.2 1B Instruct (Sambanova)",
        "family": "Llama 3.2",
        "internal_name": "Meta-Llama-3.2-1B-Instruct",
        "parameter_count": 1_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": False,
        "status": LanguageModelStatusEnum.SUBMITTED,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.2 3B Instruct (Sambanova)",
        "family": "Llama 3.2",
        "internal_name": "Meta-Llama-3.2-3B-Instruct",
        "parameter_count": 3_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": False,
        "status": LanguageModelStatusEnum.SUBMITTED,
        "semantic_group": "llama",
    },
    {
        "label": "Llama 3.3 70B Instruct (Sambanova)",
        "family": "Llama 3.3",
        "internal_name": "Meta-Llama-3.3-70B-Instruct",
        "parameter_count": 70_000_000_000,
        "context_window_tokens": 128_000,
        "is_strong": True,
        "status": LanguageModelStatusEnum.SUBMITTED,
        "semantic_group": "llama",
    },
    {
        "label": "Qwen2.5 72B Instruct (Sambanova)",
        "family": "Qwen2.5 72B",
        "internal_name": "Qwen2.5-72B-Instruct",
        "parameter_count": 72_000_000_000,
        "context_window_tokens": 131_072,
        "is_strong": True,
        "status": LanguageModelStatusEnum.SUBMITTED,
        "semantic_group": "qwen",
    },
    {
        "label": "Qwen2.5 Coder 32B Instruct (Sambanova)",
        "family": "Qwen2.5 Coder",
        "internal_name": "Qwen2.5-Coder-32B-Instruct",
        "parameter_count": 32_000_000_000,
        "context_window_tokens": 131_072,
        "is_strong": False,
        "status": LanguageModelStatusEnum.SUBMITTED,
        "semantic_group": "qwen",
    },
    {
        "label": "QwQ 32B Preview (Sambanova)",
        "family": "QwQ 32B",
        "internal_name": "QwQ-32B-Preview",
        "parameter_count": 32_000_000_000,
        "context_window_tokens": 32_768,
        "is_strong": False,
        "status": LanguageModelStatusEnum.SUBMITTED,
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
        "status": LanguageModelStatusEnum.SUBMITTED,
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
        "status": LanguageModelStatusEnum.SUBMITTED,
        "semantic_group": "llama",
    },
]

AVARTAR_URL = "https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/sambanova%20logo-DQufRZ9YknZF3bE2WoH6y0LQ2zlyuJ.webp"


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with Session(op.get_bind()) as session:
        # add organization
        org = session.exec(select(Organization).where(Organization.organization_name == "Sambanova")).first()
        if not org:
            org = Organization(organization_name="Sambanova")
            session.add(org)
            session.commit()
            session.refresh(org)
        else:
            print("Organization Sambanova already exists, skipping...")
            org.deleted_at = None
            session.add(org)
            session.commit()

        # add provider
        provider = session.exec(select(Provider).where(Provider.name == "Sambanova")).first()
        if not provider:
            provider = Provider(
                name="Sambanova",
                base_api_url="https://api.sambanova.ai/v1",
                api_key_env_name="SAMBANOVA_API_KEY",
                is_active=True,
            )            
            session.add(provider)
            session.commit()
            session.refresh(provider)
        else:
            print("Provider Sambanova already exists, skipping...")
            provider.deleted_at = None
            session.add(provider)
            session.commit()

        # add all models
        for model in SAMBANOVA_MODELS:
            existing_model = session.exec(select(LanguageModel).where(
                LanguageModel.internal_name == model["internal_name"]
            )).first()
            
            if not existing_model:
                model = LanguageModel(
                    provider_id=provider.provider_id,
                    organization_id=org.organization_id,
                    name=model["internal_name"],
                    creator_user_id="SYSTEM",
                    avatar_url=AVARTAR_URL,
                    is_pro=False,
                    **model,
                )
                session.add(model)
            else:
                print(f"Model {model['internal_name']} already exists, skipping...")
                existing_model.deleted_at = None
                session.add(existing_model)
        session.commit()

    # ### end Alembic commands ###


def downgrade() -> None:
    # # ### commands auto generated by Alembic - please adjust! ###
    # we don't delete models newly added. the upgrade() part is reentrant.
    with Session(op.get_bind()) as session:
        # soft delete all models
        for model in SAMBANOVA_MODELS:
            session.exec(
                sa.update(LanguageModel)
                .where(LanguageModel.internal_name == model["internal_name"])
                .values(deleted_at=sa.func.now())
            )

        # soft delete provider
        session.exec(
            sa.update(Provider)
            .where(Provider.name == "Sambanova")
            .values(deleted_at=sa.func.now())
        )

        # soft delete organization  
        session.exec(
            sa.update(Organization)
            .where(Organization.organization_name == "Sambanova")
            .values(deleted_at=sa.func.now())
        )
        session.commit()
    # ### end Alembic commands ###
