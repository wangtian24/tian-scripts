from typing import Any

import pytest

from ypl.backend.llm.category_labeler import (
    OFFLINE_CATEGORY,
    ONLINE_CATEGORY,
    _get_online_labeler,
    _get_topic_labeler,
)

regression_examples = [
    dict(
        content="Tell me latest news on trump inauguration",
        categories={ONLINE_CATEGORY},
    ),
    dict(
        content="All the flavors of Skittles",
        categories={OFFLINE_CATEGORY},
    ),
    dict(
        content="mountain view weather",
        categories={ONLINE_CATEGORY},
    ),
    dict(
        content="2021 major news",
        categories={OFFLINE_CATEGORY},
    ),
    dict(
        content="2023 major news",
        categories={ONLINE_CATEGORY},
    ),
    dict(
        content="all the leap years between 2000 and 2024",
        categories={OFFLINE_CATEGORY},
    ),
    dict(
        content="Does plaza premium lounge in vancouver in international departures have a tv that shows nfl games",
        categories={ONLINE_CATEGORY},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("example", regression_examples)
async def test_prompt_categorizer(example: dict[str, Any]) -> None:
    online_labeler = _get_online_labeler()
    topic_labeler = _get_topic_labeler()
    is_online = await online_labeler.alabel(example["content"])
    online_category = ONLINE_CATEGORY if is_online else OFFLINE_CATEGORY
    categories = await topic_labeler.alabel(example["content"])
    categories += [online_category]
    categories_ = set(x.lower() for x in categories)

    assert example["categories"].intersection(categories_) == example["categories"]
