"""update semantic groups

Revision ID: 0b1ce165c3d4
Revises: 2e3525fe99cf
Create Date: 2025-02-11 21:59:40.092550+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlmodel import Session

from ypl.db.language_models import LanguageModel



# revision identifiers, used by Alembic.
revision: str = '0b1ce165c3d4'
down_revision: str | None = '2e3525fe99cf'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SEMANTIC_GROUP_UPDATES = [  # (internal_name, semantic_group)
    ("ai21/jamba-1-5-large", "ai21"),
    ("ai21/jamba-1-5-mini", "ai21"),
    ("amazon/nova-pro-v1", "amazon"),
    ("claude-3-5-haiku-20241022", "claude"),
    ("claude-3-5-sonnet-20240620", "claude"),
    ("claude-3-5-sonnet-20241022", "claude"),
    ("claude-3-haiku-20240307", "claude"),
    ("claude-3-opus-20240229", "claude"),
    ("claude-3-sonnet-20240229", "claude"),
    ("cohere/command-r", "cohere"),
    ("cohere/command-r-08-2024", "cohere"),
    ("cohere/command-r-plus", "cohere"),
    ("cohere/command-r-plus-08-2024", "cohere"),
    ("databricks/dbrx-instruct", "databricks"),
    ("deepseek-coder", "deepseek"),
    ("deepseek/deepseek-chat", "deepseek"),
    ("deepseek-r1-distill-llama-70b", "deepseek r1"),
    ("deepseek/deepseek-r1", "deepseek r1"),
    ("gemini-1.5-flash-002", "gemini"),
    ("gemini-1.5-flash-8b", "gemini"),
    ("gemini-1.5-flash-8b-exp-0827", "gemini"),
    ("gemini-1.5-flash-exp-0827", "gemini"),
    ("gemini-2.0-flash-exp", "gemini"),
    ("gemini-2.0-flash-lite-preview-02-05", "gemini"),
    ("gemini-2.0-flash-thinking-exp-1219", "gemini"),
    ("gemini-exp-1114", "gemini"),
    ("gemini-1.5-pro", "gemini pro"),
    ("gemini-1.5-pro-002", "gemini pro"),
    ("gemini-1.5-pro-002-online", "gemini pro"),
    ("gemini-1.5-pro-exp-0827", "gemini pro"),
    ("gemini-2.0-pro-exp-02-05", "gemini pro"),
    ("gemma-7b-it", "gemma"),
    ("gemma2-9b-it", "gemma"),
    ("google/gemma-2-27b-it", "gemma"),
    ("google/gemma-2-9b-it", "gemma"),
    ("gpt-3.5-turbo-0125", "gpt"),
    ("gpt-4-turbo", "gpt"),
    ("gpt-4o", "gpt"),
    ("gpt-4o-2024-05-13", "gpt"),
    ("gpt-4o-2024-08-06", "gpt"),
    ("gpt-4o-mini", "gpt"),
    ("gpt-4o-mini-2024-07-18", "gpt"),
    ("o1-2024-12-17", "gpt o"),
    ("o1-mini-2024-09-12", "gpt o"),
    ("o1-preview-2024-09-12", "gpt o"),
    ("o3-mini-2025-01-31", "gpt o"),
    ("openai/o1", "gpt o"),
    ("x-ai/grok-2-1212", "grok"),
    ("x-ai/grok-beta", "grok"),
    ("gryphe/mythomax-l2-13b", "gryphe"),
    ("llama-3.1-8b-instant", "llama"),
    ("llama-3.3-70b", "llama"),
    ("llama-3.3-70b-versatile", "llama"),
    ("llama3-70b-8192", "llama"),
    ("llama3-8b-8192", "llama"),
    ("llama3.1-70b", "llama"),
    ("llama3.1-8b", "llama"),
    ("meta-llama/Llama-3-70b-chat-hf", "llama"),
    ("meta-llama/Llama-3-8b-chat-hf", "llama"),
    ("meta-llama/llama-3.1-405b-instruct", "llama"),
    ("meta-llama/llama-3.1-70b-instruct", "llama"),
    ("meta-llama/llama-3.1-8b-instruct", "llama"),
    ("meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", "llama"),
    ("meta-llama/llama-3.2-1b-instruct", "llama"),
    ("meta-llama/Llama-3.2-3B-Instruct-Turbo", "llama"),
    ("meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", "llama"),
    ("meta-llama/llama-3.3-70b-instruct", "llama"),
    ("meta-llama/Meta-Llama-3.1-70B-Instruct", "llama"),
    ("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "llama"),
    ("nousresearch/hermes-3-llama-3.1-70b", "hermes"),
    ("codestral-2405", "mistral"),
    ("ministral-3b-2410", "mistral"),
    ("ministral-8b-2410", "mistral"),
    ("mistral-large-2402", "mistral"),
    ("mistral-large-latest", "mistral"),
    ("mistral-medium", "mistral"),
    ("mistral-small-2501", "mistral"),
    ("mixtral-8x7b-32768", "mistral"),
    ("open-mistral-nemo-2407", "mistral"),
    ("open-mixtral-8x22b", "mistral"),
    ("open-mixtral-8x7b", "mistral"),
    ("pixtral-12b-2409", "mistral"),
    ("nemotron-4-340b-instruct", "nemotron"),
    ("nvidia/llama-3.1-nemotron-70b-instruct", "nemotron"),
    ("llama-3.1-sonar-huge-128k-online", "perlexity"),
    ("llama-3.1-sonar-large-128k-chat", "perlexity"),
    ("llama-3.1-sonar-large-128k-online", "perlexity"),
    ("llama-3.1-sonar-small-128k-chat", "perlexity"),
    ("llama-3.1-sonar-small-128k-online", "perlexity"),
    ("sonar", "perlexity"),
    ("sonar-pro", "perlexity"),
    ("sonar-reasoning", "perlexity"),
    ("microsoft/phi-3-medium-128k-instruct", "phi"),
    ("microsoft/phi-3-mini-4k-instruct", "phi"),
    ("microsoft/phi-3.5-mini-128k-instruct", "phi"),
    ("microsoft/phi-4", "phi"),
    ("qwen-max", "qwen"),
    ("qwen-max-2025-01-25", "qwen"),
    ("qwen-plus", "qwen"),
    ("qwen/qwen-2.5-72b-instruct", "qwen"),
    ("qwen/qwen-2.5-coder-32b-instruct", "qwen"),
    ("Qwen/Qwen1.5-110B-Chat", "qwen"),
    ("Qwen/Qwen1.5-72B-Chat", "qwen"),
    ("Qwen/Qwen2-72B-Instruct", "qwen"),
    ("qwen2.5-14b-instruct-1m", "qwen"),
    ("qwen2.5-7b-instruct-1m", "qwen"),
    ("qwen2.5-vl-3b-instruct", "qwen"),
    ("qwen2.5-vl-72b-instruct", "qwen"),
    ("qwen2.5-vl-7b-instruct", "qwen"),
    ("news-yapp", "yapp"),
    ("weather-yapp", "yapp"),
    ("wikipedia-yapp", "yapp"),
    ("youtube-transcript-yapp", "yapp"),
    ("yi-large", "yi"),
]

def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with Session(op.get_bind()) as session:
        for internal_name, semantic_group in SEMANTIC_GROUP_UPDATES:
            session.exec(sa.update(LanguageModel).where(LanguageModel.internal_name == internal_name).values(semantic_group=semantic_group))
        session.commit()
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    pass
    # ### end Alembic commands ###
