"""add llm taxo data

Revision ID: 5fb396abb721
Revises: fc9f4f641792
Create Date: 2025-02-14 19:40:17.731078+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlmodel import Session, update

from ypl.db.language_models import LanguageModel

# revision identifiers, used by Alembic.
revision: str = '5fb396abb721'
down_revision: str | None = 'fc9f4f641792'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ruff: noqa: E501
# mypy: disable-error-code=all
# The raw data is from https://docs.google.com/spreadsheets/d/1k-EhCCSxxnLWaxPFSSxRx0uA1pYJKPdCdWuYSFxqCeI/edit?gid=948246517#gid=948246517
UPDATES = [
    # internal_name, family (new value), model_class, model_version, model_release, name (new value)
    ("claude-3-haiku-20240307", "Claude", "Haiku", "3", "20240307", "anthropic/claude-3-haiku-20240307"),
    ("claude-3-5-haiku-20241022", "Claude", "Haiku", "3.5", "20241022", "anthropic/claude-3-5-haiku-20241022"),
    ("claude-3-opus-20240229", "Claude", "Opus", "3", "20240229", "anthropic/claude-3-opus-20240229"),
    ("claude-3-sonnet-20240229", "Claude", "Sonnet", "3", "", "anthropic/claude-3-sonnet-20240229"),
    ("claude-3-5-sonnet-20240620", "Claude", "Sonnet", "3.5", "20240620", "anthropic/claude-3-5-sonnet-20240620"),
    ("claude-3-5-sonnet-20241022", "Claude", "Sonnet", "3.5", "20241022", "anthropic/claude-3-5-sonnet-20241022"),
    ("cohere/command-r-08-2024", "Command", "R", "", "202408", "openrouter/cohere/command-r-08-2024"),
    ("cohere/command-r", "Command", "R", "", "", "openrouter/cohere/command-r"),
    ("cohere/command-r-plus-08-2024", "Command", "R+", "", "202408", "openrouter/cohere/command-r-plus-08-2024"),
    ("cohere/command-r-plus", "Command", "R+", "", "", "openrouter/cohere/command-r-plus"),
    ("databricks/dbrx-instruct", "DBRX", "Instruct", "", "", "openrouter/databricks/dbrx-instruct"),
    ("deepseek/deepseek-chat", "DeepSeek", "Chat", "", "", "openrouter/deepseek/deepseek-chat"),
    ("deepseek-coder", "DeepSeek", "Coder", "", "", "deepseek/deepseek-coder"),
    ("deepseek-r1-distill-llama-70b", "DeepSeek", "R1 distill llama 70b", "", "", "groq/deepseek-r1-distill-llama-70b"),
    ("DeepSeek-R1-Distill-Llama-70B", "DeepSeek", "R1 distill llama 70b", "", "", "sambanova/DeepSeek-R1-Distill-Llama-70B"),
    ("deepseek/deepseek-r1", "DeepSeek", "R1", "", "", "openrouter/deepseek/deepseek-r1"),
    ("gemini-exp-1114", "Gemini", "", "", "20241114", "google/gemini-exp-1114"),
    ("gemini-1.5-flash-8b-exp-0827", "Gemini", "Flash 8b", "1.5", "20240827", "google/gemini-1.5-flash-8b-exp-0827"),
    ("gemini-1.5-flash-8b", "Gemini", "Flash 8b", "1.5", "", "google/gemini-1.5-flash-8b"),
    ("gemini-2.0-flash-lite-preview-02-05", "Gemini", "Flash Lite", "2.0", "", "google/gemini-2.0-flash-lite-preview-02-05"),
    ("gemini-2.0-flash-thinking-exp-1219", "Gemini", "Flash Thinking", "2.0", "", "google/gemini-2.0-flash-thinking-exp-1219"),
    ("gemini-1.5-flash-002", "Gemini", "Flash", "1.5", "002", "google/gemini-1.5-flash-002"),
    ("gemini-1.5-flash-exp-0827", "Gemini", "Flash", "1.5", "20240827", "google/gemini-1.5-flash-exp-0827"),
    ("gemini-2.0-flash-exp", "Gemini", "Flash", "2.0", "", "google/gemini-2.0-flash-exp"),
    ("gemini-1.5-pro", "Gemini", "Pro", "1.5", "", "google/gemini-1.5-pro"),
    ("gemini-1.5-pro-002", "Gemini", "Pro", "1.5", "002", "google/gemini-1.5-pro-002"),
    ("gemini-1.5-pro-002-online", "Gemini", "Pro online", "1.5", "", "googlegrounded/gemini-1.5-pro-002-online"),
    ("gemini-1.5-pro-exp-0827", "Gemini", "Pro", "1.5", "20240827", "google/gemini-1.5-pro-exp-0827"),
    ("gemini-2.0-pro-exp-02-05", "Gemini", "Pro", "2.0", "20250205", "google/gemini-2.0-pro-exp-02-05"),
    ("google/gemma-2-27b-it", "Gemma", "27b", "2", "", "together_ai/google/gemma-2-27b-it"),
    ("gemma-7b-it", "Gemma", "7b", "", "", "groq/gemma-7b-it"),
    ("google/gemma-2-9b-it", "Gemma", "9b", "", "", "together_ai/google/gemma-2-9b-it"),
    ("gemma2-9b-it", "Gemma", "9b", "2", "", "groq/gemma2-9b-it"),
    ("gpt-4o-mini", "GPT", "4o mini", "", "", "openai/gpt-4o-mini"),
    ("gpt-4o-mini-2024-07-18", "GPT", "4o mini", "", "", "openai/gpt-4o-mini-2024-07-18"),
    ("gpt-4o", "GPT", "4o", "", "", "openai/gpt-4o"),
    ("gpt-4o-2024-05-13", "GPT", "4o", "", "", "openai/gpt-4o-2024-05-13"),
    ("gpt-4o-2024-08-06", "GPT", "4o", "", "", "openai/gpt-4o-2024-08-06"),
    ("o1-mini-2024-09-12", "GPT", "o1 mini", "", "20240912", "openai/o1-mini-2024-09-12"),
    ("o1-preview-2024-09-12", "GPT", "o1 preview", "", "20240912", "openai/o1-preview-2024-09-12"),
    ("o1-2024-12-17", "GPT", "o1", "", "20241217", "openai/o1-2024-12-17"),
    ("openai/o1", "GPT", "o1", "", "", "openrouter/openai/o1"),
    ("o3-mini-2025-01-31", "GPT", "o3 mini", "", "20250131", "openai/o3-mini-2025-01-31"),
    ("gpt-3.5-turbo-0125", "GPT", "turbo", "3.5", "20240125", "openai/gpt-3.5-turbo-0125"),
    ("gpt-4-turbo", "GPT", "turbo", "4", "", "openai/gpt-4-turbo"),
    ("x-ai/grok-2-1212", "Grok", "", "2", "20241212", "openrouter/x-ai/grok-2-1212"),
    ("x-ai/grok-beta", "Grok", "", "beta", "", "openrouter/x-ai/grok-beta"),
    ("nousresearch/hermes-3-llama-3.1-70b", "Hermes", "llama 3.1 70b", "3", "", "openrouter/nousresearch/hermes-3-llama-3.1-70b"),
    ("ai21/jamba-1-5-large", "Jamba", "Large", "1.5", "", "openrouter/ai21/jamba-1-5-large"),
    ("ai21/jamba-1-5-mini", "Jamba", "Mini", "1.5", "", "openrouter/ai21/jamba-1-5-mini"),
    ("meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", "Llama", "11b vision instruct", "3.2", "", "together_ai/meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"),
    ("Llama-3.2-11B-Vision-Instruct", "Llama", "11b vision instruct", "3.2", "", "sambanova/Llama-3.2-11B-Vision-Instruct"),
    ("meta-llama/llama-3.2-1b-instruct", "Llama", "1b instruct", "3.2", "", "openrouter/meta-llama/llama-3.2-1b-instruct"),
    ("Meta-Llama-3.2-1B-Instruct", "Llama", "1b instruct", "3.2", "", "sambanova/Meta-Llama-3.2-1B-Instruct"),
    ("meta-llama/Llama-3.2-3B-Instruct-Turbo", "Llama", "3b instruct turbo", "3.2", "", "together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo"),
    ("Meta-Llama-3.2-3B-Instruct", "Llama", "3b instruct", "3.2", "", "sambanova/Meta-Llama-3.2-3B-Instruct"),
    ("Meta-Llama-3.1-405B-Instruct", "Llama", "405b instruct", "3.1", "", "sambanova/Meta-Llama-3.1-405B-Instruct"),
    ("meta-llama/llama-3.1-405b-instruct", "Llama", "405b instruct", "3.1", "", "openrouter/meta-llama/llama-3.1-405b-instruct"),
    ("meta-llama/Llama-3-70b-chat-hf", "Llama", "70b chat hf", "3", "", "together_ai/meta-llama/Llama-3-70b-chat-hf"),
    ("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "Llama", "70b instruct turbo", "3.1", "", "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    ("meta-llama/llama-3.1-70b-instruct", "Llama", "70b instruct", "3.1", "", "openrouter/meta-llama/llama-3.1-70b-instruct"),
    ("meta-llama/Meta-Llama-3.1-70B-Instruct", "Llama", "70b instruct", "3.1", "", "anyscale/meta-llama/Meta-Llama-3.1-70B-Instruct"),
    ("Meta-Llama-3.1-70B-Instruct", "Llama", "70b instruct", "3.1", "", "sambanova/Meta-Llama-3.1-70B-Instruct"),
    ("meta-llama/llama-3.3-70b-instruct", "Llama", "70b instruct", "3.3", "", "openrouter/meta-llama/llama-3.3-70b-instruct"),
    ("Meta-Llama-3.3-70B-Instruct", "Llama", "70b instruct", "3.3", "", "sambanova/Meta-Llama-3.3-70B-Instruct"),
    ("llama-3.3-70b-versatile", "Llama", "70b versatile", "3.3", "", "groq/llama-3.3-70b-versatile"),
    ("llama3-70b-8192", "Llama", "70b", "3", "", "groq/llama3-70b-8192"),
    ("llama3.1-70b", "Llama", "70b", "3.1", "", "cerebras/llama3.1-70b"),
    ("llama-3.3-70b", "Llama", "70b", "3.3", "", "cerebras/llama-3.3-70b"),
    ("meta-llama/Llama-3-8b-chat-hf", "Llama", "8b chat hf", "3", "", "together_ai/meta-llama/Llama-3-8b-chat-hf"),
    ("llama-3.1-8b-instant", "Llama", "8b instant", "3.1", "", "groq/llama-3.1-8b-instant"),
    ("meta-llama/llama-3.1-8b-instruct", "Llama", "8b instruct", "3.1", "", "openrouter/meta-llama/llama-3.1-8b-instruct"),
    ("Meta-Llama-3.1-8B-Instruct", "Llama", "8b instruct", "3.1", "", "sambanova/Meta-Llama-3.1-8B-Instruct"),
    ("llama3-8b-8192", "Llama", "8b", "3", "", "groq/llama3-8b-8192"),
    ("llama3.1-8b", "Llama", "8b", "3.1", "", "cerebras/llama3.1-8b"),
    ("meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", "Llama", "90b vision instruct turbo", "3.2", "", "together_ai/meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"),
    ("Llama-3.2-90B-Vision-Instruct", "Llama", "90b vision instruct", "3.2", "", "sambanova/Llama-3.2-90B-Vision-Instruct"),
    ("Llama-3.1-Tulu-3-405B", "Llama", "tulu 3 405b", "3.1", "", "sambanova/Llama-3.1-Tulu-3-405B"),
    ("codestral-2405", "Mistral", "Codestral", "", "202405", "mistral_ai/codestral-2405"),
    ("mistral-large-2402", "Mistral", "Large", "", "202402", "mistral_ai/mistral-large-2402"),
    ("mistral-large-latest", "Mistral", "Large", "", "latest", "mistral_ai/mistral-large-latest"),
    ("mistral-medium", "Mistral", "Medium", "", "", "mistral_ai/mistral-medium"),
    ("ministral-3b-2410", "Mistral", "Ministral 3b", "", "202410", "mistral_ai/ministral-3b-2410"),
    ("ministral-8b-2410", "Mistral", "Ministral 8b", "", "202410", "mistral_ai/ministral-8b-2410"),
    ("open-mixtral-8x22b", "Mistral", "Mixtral 8x22b", "", "", "mistral_ai/open-mixtral-8x22b"),
    ("mixtral-8x7b-32768", "Mistral", "Mixtral 8x7b", "", "", "groq/mixtral-8x7b-32768"),
    ("open-mixtral-8x7b", "Mistral", "Mixtral 8x7b", "", "", "mistral_ai/open-mixtral-8x7b"),
    ("open-mistral-nemo-2407", "Mistral", "Nemo", "", "202407", "mistral_ai/open-mistral-nemo-2407"),
    ("pixtral-12b-2409", "Mistral", "Pixtral 12b", "", "202409", "mistral_ai/pixtral-12b-2409"),
    ("mistral-small-2501", "Mistral", "Small", "", "202501", "mistral_ai/mistral-small-2501"),
    ("gryphe/mythomax-l2-13b", "MythoMax", "L2 13b", "", "", "openrouter/gryphe/mythomax-l2-13b"),
    ("nemotron-4-340b-instruct", "Nemotron", "340b instruct", "4", "", "nvidia/nemotron-4-340b-instruct"),
    ("nvidia/llama-3.1-nemotron-70b-instruct", "Nemotron", "70b instruct", "3.1", "", "openrouter/nvidia/llama-3.1-nemotron-70b-instruct"),
    ("amazon/nova-pro-v1", "Nova", "Pro", "1", "", "openrouter/amazon/nova-pro-v1"),
    ("microsoft/phi-4", "Phi", "", "4", "", "openrouter/microsoft/phi-4"),
    ("microsoft/phi-3-medium-128k-instruct", "Phi", "medium 128k instruct", "3", "", "openrouter/microsoft/phi-3-medium-128k-instruct"),
    ("microsoft/phi-3.5-mini-128k-instruct", "Phi", "mini 128k instruct", "3.5", "", "openrouter/microsoft/phi-3.5-mini-128k-instruct"),
    ("microsoft/phi-3-mini-4k-instruct", "Phi", "mini 4k instruct", "3", "", "hugging_face/microsoft/phi-3-mini-4k-instruct"),
    ("Qwen/Qwen1.5-110B-Chat", "Qwen", "110b chat", "1.5", "", "together_ai/Qwen/Qwen1.5-110B-Chat"),
    ("qwen2.5-14b-instruct-1m", "Qwen", "14b instruct", "2.5", "", "alibaba/qwen2.5-14b-instruct-1m"),
    ("Qwen/Qwen1.5-72B-Chat", "Qwen", "72b chat", "1.5", "", "together_ai/Qwen/Qwen1.5-72B-Chat"),
    ("Qwen/Qwen2-72B-Instruct", "Qwen", "72b instruct", "2", "", "together_ai/Qwen/Qwen2-72B-Instruct"),
    ("qwen/qwen-2.5-72b-instruct", "Qwen", "72b instruct", "2.5", "", "openrouter/qwen/qwen-2.5-72b-instruct"),
    ("qwen2.5-7b-instruct-1m", "Qwen", "7b instruct", "2.5", "", "alibaba/qwen2.5-7b-instruct-1m"),
    ("qwen/qwen-2.5-coder-32b-instruct", "Qwen", "coder 32b instruct", "2.5", "", "openrouter/qwen/qwen-2.5-coder-32b-instruct"),
    ("qwen-max-2025-01-25", "Qwen", "max", "", "20250125", "alibaba/qwen-max-2025-01-25"),
    ("qwen-max", "Qwen", "max", "", "", "alibaba/qwen-max"),
    ("qwen-plus", "Qwen", "plus", "", "", "alibaba/qwen-plus"),
    ("qwen2.5-vl-3b-instruct", "Qwen", "vl 3b instruct", "2.5", "", "alibaba/qwen2.5-vl-3b-instruct"),
    ("qwen2.5-vl-72b-instruct", "Qwen", "vl 72b instruct", "2.5", "", "alibaba/qwen2.5-vl-72b-instruct"),
    ("qwen2.5-vl-7b-instruct", "Qwen", "vl 7b instruct", "2.5", "", "alibaba/qwen2.5-vl-7b-instruct"),
    ("QwQ-32B-Preview", "Qwen", "QwQ 32b preview", "", "", "sambanova/QwQ-32B-Preview"),
    ("sonar", "Sonar", "", "", "", "perplexity/sonar"),
    ("llama-3.1-sonar-huge-128k-online", "Sonar", "huge online", "", "", "perplexity/llama-3.1-sonar-huge-128k-online"),
    ("llama-3.1-sonar-large-128k-chat", "Sonar", "large chat", "", "", "perplexity/llama-3.1-sonar-large-128k-chat"),
    ("llama-3.1-sonar-large-128k-online", "Sonar", "large online", "", "", "perplexity/llama-3.1-sonar-large-128k-online"),
    ("sonar-pro", "Sonar", "Pro", "", "", "perplexity/sonar-pro"),
    ("sonar-reasoning", "Sonar", "Reasoning", "", "", "perplexity/sonar-reasoning"),
    ("llama-3.1-sonar-small-128k-chat", "Sonar", "small chat", "", "", "perplexity/llama-3.1-sonar-small-128k-chat"),
    ("llama-3.1-sonar-small-128k-online", "Sonar", "small online", "", "", "perplexity/llama-3.1-sonar-small-128k-online"),
    ("news-yapp", "Yapp", "News", "", "", "yapp_temporary/news-yapp"),
    ("weather-yapp", "Yapp", "Weather", "", "", "yapp_temporary/weather-yapp"),
    ("wikipedia-yapp", "Yapp", "Wikipedia", "", "", "yapp_temporary/wikipedia-yapp"),
    ("youtube-transcript-yapp", "Yapp", "YouTube Transcript", "", "", "yapp_temporary/youtube-transcript-yapp"),
    ("yi-large", "Yi", "Large", "", "", "nvidia/yi-large"),    
]
# mypy: enable-error-code=all
# ruff: enable


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # internal_name, family (new value), model_class, model_version, model_release, name (new value)
    with Session(op.get_bind()) as session:
        for internal_name, family, model_class, model_version, model_release, name in UPDATES:
            session.exec(sa.update(LanguageModel)
                         .where(LanguageModel.internal_name == internal_name)
                         .values(family=family, model_class=model_class, model_version=model_version,
                                model_release=model_release, name=name))
        session.commit()
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    pass
    # ### end Alembic commands ###
