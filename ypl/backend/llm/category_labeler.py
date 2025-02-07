import asyncio

from ypl.backend.config import settings
from ypl.backend.llm.constants import OFFLINE_CATEGORY, ONLINE_CATEGORY
from ypl.backend.llm.judge import YuppMultilabelClassifier, YuppOnlinePromptLabeler
from ypl.backend.llm.routing.router import _get_routing_llm

ONLINE_LABELER: YuppOnlinePromptLabeler | None = None
TOPIC_LABELER: YuppMultilabelClassifier | None = None


async def get_prompt_categories(prompt: str) -> list[str]:
    # Labeling the prompt
    online_labeler = _get_online_labeler()
    topic_labeler = _get_topic_labeler()
    online_label, topic_labels = await asyncio.gather(
        online_labeler.alabel(prompt),
        topic_labeler.alabel(prompt),
    )
    online_category = ONLINE_CATEGORY if online_label else OFFLINE_CATEGORY
    categories = [online_category] + topic_labels
    categories = list(dict.fromkeys(categories))
    return categories


def _get_online_labeler() -> YuppOnlinePromptLabeler:
    global ONLINE_LABELER
    if ONLINE_LABELER is None:
        ONLINE_LABELER = YuppOnlinePromptLabeler(_get_routing_llm(), timeout_secs=settings.ROUTING_TIMEOUT_SECS)
    return ONLINE_LABELER


def _get_topic_labeler() -> YuppMultilabelClassifier:
    global TOPIC_LABELER
    if TOPIC_LABELER is None:
        TOPIC_LABELER = YuppMultilabelClassifier(_get_routing_llm(), timeout_secs=settings.ROUTING_TIMEOUT_SECS)
    return TOPIC_LABELER
