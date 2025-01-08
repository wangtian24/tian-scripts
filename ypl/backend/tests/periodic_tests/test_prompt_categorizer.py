from typing import Any

import pytest

from ypl.backend.llm.routing.router import get_online_labeler, get_topic_labeler

regression_examples = [
    dict(
        content="Tell me latest news on trump inauguration",
        categories={"online"},
    ),
    dict(
        content="All the flavors of Skittles",
        categories={"offline"},
    ),
    dict(
        content="mountain view weather",
        categories={"online"},
    ),
    dict(
        content="2021 major news",
        categories={"offline"},
    ),
    dict(
        content="2023 major news",
        categories={"online"},
    ),
    dict(
        content="all the leap years between 2000 and 2024",
        categories={"offline"},
    ),
    dict(
        content="Does plaza premium lounge in vancouver in international departures have a tv that shows nfl games",
        categories={"online"},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("example", regression_examples)
async def test_prompt_categorizer(example: dict[str, Any]) -> None:
    online_labeler = get_online_labeler()
    topic_labeler = get_topic_labeler()
    is_online = await online_labeler.alabel(example["content"])
    online_category = "online" if is_online else "offline"
    categories = await topic_labeler.alabel(example["content"])
    categories += [online_category]
    categories_ = set(x.lower() for x in categories)

    assert example["categories"].intersection(categories_) == example["categories"]
