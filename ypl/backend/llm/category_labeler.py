import asyncio

from langchain_core.language_models.chat_models import BaseChatModel

from ypl.backend.config import settings
from ypl.backend.llm.constants import OFFLINE_CATEGORY, ONLINE_CATEGORY
from ypl.backend.llm.judge import YuppMultilabelClassifier, YuppOnlinePromptLabeler
from ypl.backend.llm.provider.provider_clients import get_provider_client
from ypl.backend.llm.routing.router import _get_routing_llm

ONLINE_LABELER: YuppOnlinePromptLabeler | None = None
TOPIC_LABELER: YuppMultilabelClassifier | None = None

CATEGORY_LABELING_MODEL = "gemini-1.5-flash-8b"


def merge_categories(topic_labels: list[str], online_label: bool) -> list[str]:
    online_category = ONLINE_CATEGORY if online_label else OFFLINE_CATEGORY
    categories = [online_category] + topic_labels
    return list(dict.fromkeys(categories))


async def get_prompt_categories(prompt: str) -> list[str]:
    # Labeling the prompt
    online_labeler = await _get_online_labeler()
    topic_labeler = await _get_topic_labeler()
    online_label, topic_labels = await asyncio.gather(
        online_labeler.alabel(prompt),
        topic_labeler.alabel(prompt),
    )
    categories = merge_categories(topic_labels, online_label)
    return categories


async def _get_online_labeler() -> YuppOnlinePromptLabeler:
    global ONLINE_LABELER
    if ONLINE_LABELER is None:
        ONLINE_LABELER = YuppOnlinePromptLabeler(
            await get_prompt_online_classifier_llm(CATEGORY_LABELING_MODEL),
            timeout_secs=settings.ROUTING_TIMEOUT_SECS,
        )
    return ONLINE_LABELER


async def _get_topic_labeler() -> YuppMultilabelClassifier:
    global TOPIC_LABELER
    if TOPIC_LABELER is None:
        TOPIC_LABELER = YuppMultilabelClassifier(
            await get_prompt_category_classifier_llm(CATEGORY_LABELING_MODEL),
            timeout_secs=settings.ROUTING_TIMEOUT_SECS,
        )
    return TOPIC_LABELER


async def get_prompt_category_classifier_llm(model_name: str | None = None) -> BaseChatModel:
    if model_name:
        return await get_provider_client(model_name)
    else:
        return _get_routing_llm()


async def get_prompt_online_classifier_llm(model_name: str | None = None) -> BaseChatModel:
    if model_name:
        return await get_provider_client(model_name)
    else:
        return _get_routing_llm()
