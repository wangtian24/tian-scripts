import logging
import uuid
from collections import Counter
from datetime import UTC, datetime

from ypl.backend.db import get_async_session
from ypl.backend.llm.searchapi import fetch_search_api_response
from ypl.backend.utils.json import json_dumps
from ypl.db.trends import NewsStory, TrendingTopic

# Locations to fetch trending topics for.
TRENDING_TOPICS_GEOS = ("US", "IN")

# The maximum number of trends per category ("business", "entertainment", etc.).
MAX_TRENDS_PER_CATEGORY = 3

# The maximum number of keywords stored per trend.
MAX_KEYWORDS_PER_TREND = 5

# The default categories to assign to a trend, if no categories are provided.
DEFAULT_CATEGORIES = ["general"]


def _parse_iso_date(date_str: str | None) -> datetime | None:
    """Helper because some dates are not timezone-aware."""
    if not date_str:
        return None

    try:
        date = datetime.fromisoformat(date_str)
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        return date
    except Exception as e:
        logging.error(f"Error parsing date {date_str}: {e}")
        return None


async def get_google_trends_trending_now(geo: str = "US", time: str = "past_4_hours") -> dict:
    return await fetch_search_api_response(
        {
            "engine": "google_trends_trending_now",
            "geo": geo,
            "time": time,
        }
    )


async def get_google_trends_trending_now_news(news_token: str) -> dict:
    return await fetch_search_api_response(
        {
            "engine": "google_trends_trending_now_news",
            "news_token": news_token,
        }
    )


async def update_trending_topics(geo: str = "US") -> list[uuid.UUID]:
    """Fetch recent trending topics and their related news stories, update the DB, and return the trending topic IDs."""
    try:
        trends_response = await get_google_trends_trending_now(geo=geo)
    except Exception as e:
        logging.error(f"Error fetching trends: {e}")
        return []

    trends_data = trends_response.get("trends", [])
    logging.info(f"Fetched {len(trends_data)} trends for geo {geo} from Google Trends")

    # Keep counts of how many trends we have for each category, to cap the number of trends per category.
    category_counts: Counter[str] = Counter()

    # The generated trending topics and their news stories.
    trends: list[TrendingTopic] = []
    news_stories: list[NewsStory] = []

    for trend_data in trends_data:
        categories = trend_data.get("categories", DEFAULT_CATEGORIES)
        for category in categories:
            category_counts[category] += 1

        # If all categories for this trend have reached the limit, skip it.
        trend_category_counts = [category_counts[category] for category in categories]
        if all(count > MAX_TRENDS_PER_CATEGORY for count in trend_category_counts):
            continue

        trend = TrendingTopic(
            trending_topic_id=uuid.uuid4(),
            position=trend_data.get("position", 0),
            query=trend_data.get("query", ""),
            location=trend_data.get("location", geo),
            categories=categories,
            start_date=_parse_iso_date(trend_data.get("start_date")),
            end_date=_parse_iso_date(trend_data.get("end_date")),
            is_active=trend_data.get("is_active"),
            keywords=trend_data.get("keywords", [])[:MAX_KEYWORDS_PER_TREND],
        )

        # Fetch the news stories for the trending topic.
        news_token = trend_data.get("news_token")
        if not news_token:
            logging.warning(f"Trend {trend_data.get('query')} has no news token")
            continue

        news_items: list[dict] = []
        try:
            news_response = await get_google_trends_trending_now_news(news_token)
        except Exception as e:
            logging.error(f"Error fetching news for trend {trend_data.get('query')}: {e}")
            continue

        news_items = news_response.get("news", [])
        trend_news_stories = []
        for news_item in news_items:
            news_story = NewsStory(
                title=news_item.get("title", ""),
                link=news_item.get("link", ""),
                source=news_item.get("source", ""),
                date=_parse_iso_date(news_item["iso_date"]),
                thumbnail=news_item.get("thumbnail"),
                trending_topic_id=trend.trending_topic_id,
            )
            trend_news_stories.append(news_story)

        logging.info(
            json_dumps(
                {
                    "message": f"Updated trending topic '{trend.query}' for geo '{geo}'",
                    "trend": trend.model_dump(mode="json", exclude_unset=True),
                    "news_stories": [
                        news_story.model_dump(mode="json", exclude_unset=True) for news_story in trend_news_stories
                    ],
                }
            )
        )
        trends.append(trend)
        news_stories.extend(trend_news_stories)

    async with get_async_session() as session:
        session.add_all(trends)
        session.add_all(news_stories)
        await session.commit()

    return [trend.trending_topic_id for trend in trends]
