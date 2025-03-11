"""Tests for the YAPP agent classifier functionality."""

from typing import Any

import pytest

from ypl.backend.llm.category_labeler import _get_yapp_matcher

# Examples for testing the YAPP agent classifier
yapp_classifier_examples = [
    dict(
        content="What's the current temperature in Vancouver?",
        expected_label="weather-yapp",
    ),
    dict(
        content="Will it rain in Miami today or is it sunny?",
        expected_label="weather-yapp",
    ),
    dict(
        content="Compare the current weather conditions between Paris and London",
        expected_label="weather-yapp",
    ),
    dict(
        content="Do I need an umbrella in Seattle given the weather forecast?",
        expected_label="weather-yapp",
    ),
    dict(
        content="What was the average rainfall trends in Tokyo during 2020?",
        expected_label="none",
    ),
    dict(
        content="Tell me about the climate patterns in the Amazon rainforest",
        expected_label="none",
    ),
    dict(
        content="What are the breaking developments in the Middle East peace talks from todays news?",
        expected_label="news-yapp",
    ),
    dict(
        content="How are the financial markets performing today?",
        expected_label="news-yapp",
    ),
    dict(
        content="What were the outcomes of today's climate summit?",
        expected_label="news-yapp",
    ),
    dict(
        content="Share the latest news from the ongoing BRICS summit",
        expected_label="news-yapp",
    ),
    dict(
        content="Write a fanfiction about the 1980 presidential election",
        expected_label="none",
    ),
    dict(
        content="Tell me a story about the 1980 presidential election?",
        expected_label="none",
    ),
    dict(
        content="who was the first president of the United States?",
        expected_label="wikipedia-yapp",
    ),
    dict(
        content="name of the scientist who discovered the photoelectric effect",
        expected_label="wikipedia-yapp",
    ),
    dict(
        content="Explain the basic principles of quantum mechanics",
        expected_label="none",
    ),
    dict(
        content="Write about the history of the Roman Empire in a poetic style",
        expected_label="none",
    ),
    dict(
        content="Summarize what's spoken in this video in the conclusion please: https://youtube.com/watch?v=abc123",
        expected_label="youtube-transcript-yapp",
    ),
    dict(
        content="https://youtu.be/xyz789 - describe what the speaker is getting to at 1:42"
        "and tell me what they said at 2:30",
        expected_label="youtube-transcript-yapp",
    ),
    dict(
        content="Key ideas discussed in this video: https://youtube.com/watch?v=def456",
        expected_label="youtube-transcript-yapp",
    ),
    dict(
        content="What did they cover in this youtube videos: https://www.youtube.com/watch?v=Uwmp16aSgdk, https://www.youtube.com/watch?v=abc123",
        expected_label="youtube-transcript-yapp",
    ),
    dict(
        content="What color shirts are the people in this video wearing: https://www.youtube.com/watch?v=Uwmp16aSgdk.",
        expected_label="none",
    ),
    dict(
        content="What's the latest viral video on YouTube?",
        expected_label="none",
    ),
    dict(
        content="Can you find me some good programming tutorials?",
        expected_label="none",
    ),
    dict(
        content="""A researcher has evidence that a rare molecule found in a species of tropical plant, in low doses,
target a parasite that infects human dermal tissue. However, in order to prove his hypothesis, he needs more than his
current circumstantial evidence of parasite death. For instance, the parasite infection may have cleared because of
a stimulated immune system, random chance, a genetic resistance, etc. The researcher needs to determine the
mode of action of the tropical plant drug. He hypothesizes that the drug (ligand) irreversibly binds to some
protein in the parasite. Proving the hypothesis and demonstrating a method of action would improve the
likelihood that human test trials would be approved. The researcher wonders how many techniques would be
needed to verify the protein and protein-binding site. What is the reason for the parasite infection?""",
        expected_label="none",
    ),
    dict(
        content="""Debugging a code that is not working.
The code is a python script that is not working.
The error message is: 'NameError: name 'x' is not defined'
The code is:
```python
def add(a, b):
    return a + b

print(add(1, 2, 3))
```
""",
        expected_label="none",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("example", yapp_classifier_examples)
async def test_yapp_classifier(example: dict[str, Any]) -> None:
    """Test the YAPP agent classifier function."""
    classifier = await _get_yapp_matcher()
    result = await classifier.alabel(example["content"])
    assert (
        result == example["expected_label"]
    ), f"Expected {example['expected_label']}, got {result} for '{example['content']}'"
