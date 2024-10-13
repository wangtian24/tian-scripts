import datetime
import logging
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

import numpy as np
import tweepy
from slack_sdk.webhook import WebhookClient
from sqlmodel import select

from ypl.backend.db import Session, get_engine
from ypl.db.ratings import OVERALL_CATEGORY_NAME, Category

EPSILON = 1e-9


def combine_short_sentences(
    sentences: list[str], max_combined_length: int = 40, max_single_length: int = 10
) -> list[str]:
    """Combine short sentences into a single sentence."""
    combined_sentences = []
    current_sentence = ""

    for sentence in sentences:
        if (len(current_sentence) + len(sentence) <= max_combined_length) or (
            len(current_sentence) < max_single_length
        ):
            current_sentence += " " + sentence if current_sentence else sentence
        else:
            if current_sentence:
                combined_sentences.append(current_sentence.strip())
            current_sentence = sentence

    if current_sentence:
        combined_sentences.append(current_sentence.strip())

    return combined_sentences


@dataclass
class Battle:
    model_a: str
    model_b: str
    # Convention is between [0..1], where 0 means "loss" and 1 means "win".
    result_a: float

    def winner(self) -> str | None:
        return self.model_a if self.result_a > 0.5 else self.model_b if self.result_a < 0.5 else None

    def loser(self) -> str | None:
        return self.model_b if self.result_a > 0.5 else self.model_a if self.result_a < 0.5 else None

    def tie(self) -> bool:
        return self.result_a == 0.5


@dataclass
class AnnotatedFloat:
    """An annotated value."""

    value: float | None
    annotation: str | None

    def __float__(self) -> float | None:
        return self.value


@dataclass
class RatedModel:
    """A model with a rating."""

    model: str
    rating: float
    rating_lower: float | None = None
    rating_upper: float | None = None
    wins: int = 0
    losses: int = 0
    ties: int = 0
    annotation: str | None = None


def norm_softmax(arr: Iterable[float]) -> np.ndarray:
    """
    Returns pseudo-probabilities using a sigmoid-like normalization and softmax.
    """
    arr = np.array(arr, dtype=float)

    if any(np.isnan(x) or np.isinf(x) for x in arr):
        raise ValueError("Input array contains infinite or NaN values.")

    if len(arr) <= 1:
        return np.array([1] * len(arr))

    if np.all(arr == arr[0]):
        return np.full_like(arr, 1 / len(arr))  # Uniform distribution

    scale = max(np.abs(arr).max(), EPSILON)
    sigmoid = 1 / (1 + np.exp(-arr / scale))
    exp_sigmoid = np.exp(sigmoid)

    return np.array(exp_sigmoid / np.sum(exp_sigmoid))


@dataclass
class ThresholdCounter:
    """A counter that tracks updates and determines when a threshold is reached."""

    # Number of times the counter was incremented since last reset.
    count: int = 0
    # Total number of times the counter was incremented.
    total_count: int = 0
    # The threshold at which the count is considered to be reached.
    threshold: int = 1
    # The maximum threshold value.
    max_threshold: int = 25000
    # The rate at which the threshold grows every time it is reset.
    growth_rate: float = 1.2345
    # Number of times the threshold was reset.
    reset_count: int = 0

    def increment(self) -> None:
        self.count += 1
        self.total_count += 1

    def is_threshold_reached(self) -> bool:
        return self.count >= self.threshold

    def reset(self) -> None:
        """Resets the counter and increases the threshold."""
        self.count = 0
        self.threshold = min(math.ceil(self.threshold * self.growth_rate), self.max_threshold)
        self.reset_count += 1


@cache
def fetch_categories_with_descriptions_from_db() -> dict[str, str | None]:
    """Returns a mapping between category names and their descriptions, fetched from the database."""
    with Session(get_engine()) as session:
        categories = session.exec(
            select(Category.name, Category.description).where(Category.name != OVERALL_CATEGORY_NAME)
        ).all()
        return {name: description for name, description in categories}


async def post_to_slack(message: str) -> None:
    """
    Post a message to a Slack channel using a webhook URL.

    Args:
        message (str): The message to post to Slack.
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logging.warning("SLACK_WEBHOOK_URL environment variable is not set")
        return

    try:
        webhook = WebhookClient(webhook_url)
        response = webhook.send(text=message)
        if response.status_code != 200:
            logging.warning(f"Failed to post message to Slack. Status code: {response.status_code}")
    except Exception as e:
        logging.warning(f"Failed to post message to Slack: {str(e)}")


async def post_to_x(message: str) -> None:
    """Post a message to X."""
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    consumer_key = os.environ.get("TWITTER_CONSUMER_KEY")
    consumer_secret = os.environ.get("TWITTER_CONSUMER_SECRET")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
    if bearer_token and consumer_key and consumer_secret and access_token and access_token_secret:
        try:
            client = tweepy.Client(
                bearer_token=bearer_token,
                access_token=access_token,
                access_token_secret=access_token_secret,
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
            )
            # TODO: Use some random number to avoid duplicate tweets.
            # Uncomment the following line to tweet the actual message.
            # tweet = datetime.datetime.now(datetime.UTC).isoformat() + " " + message
            tweet = datetime.datetime.now(datetime.UTC).isoformat() + " " + "Hello...something magical just happened!"
            client.create_tweet(text=tweet)
        except Exception as e:
            logging.warning(f"Failed to post message to X: {str(e)}")
    else:
        logging.warning(
            "TWITTER_BEARER_TOKEN, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, or "
            "TWITTER_ACCESS_TOKEN_SECRET environment variable is not set"
        )
