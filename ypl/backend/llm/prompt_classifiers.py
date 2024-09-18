from enum import Enum  # noqa: I001
from functools import cache
from typing import Any, cast

import logging

from openai import OpenAI
from pydantic import BaseModel
from sqlmodel import select
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ypl.backend import prompts
from ypl.backend.db import Session, get_engine
from ypl.backend.llm.utils import fetch_categories_with_descriptions_from_db
from ypl.db.chats import ChatMessage, MessageType
from ypl.db.ratings import Category, OVERALL_CATEGORY_NAME


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


client: OpenAI | None = None


def initialize_client(**kwargs: Any) -> OpenAI:
    global client
    if client is None:
        client = OpenAI(**kwargs)
    return client


def construct_system_prompt(system_prompt: str, category_descriptions_dict: dict[str, str]) -> str:
    formatted_categories = [f"- {name}: {description}" for name, description in category_descriptions_dict.items()]
    return system_prompt.replace("<REPLACE_WITH_CATEGORIES_FROM_DB>", "\n".join(formatted_categories))


class PromptCategoryResponse(BaseModel):
    category: Enum


@cache
def llm_setup() -> tuple[str, type[PromptCategoryResponse]]:
    category_descriptions_dict = {
        category: description
        for category, description in fetch_categories_with_descriptions_from_db().items()
        if description is not None
    }

    system_prompt = construct_system_prompt(prompts.PROMPT_CATEGORY_SYSTEM_TEMPLATE, category_descriptions_dict)
    PromptCategory = Enum(  # type: ignore
        "PromptCategory",
        {category.replace(" ", "_").upper(): category for category in category_descriptions_dict.keys()},
    )

    class _PromptCategoryResponse(PromptCategoryResponse):
        category: PromptCategory

    return system_prompt, _PromptCategoryResponse


@retry(wait=wait_random_exponential(), stop=stop_after_attempt(3))
def prompt_category_by_llm(user_prompt: str) -> str:
    client = initialize_client()

    system_prompt, PromptCategoryResponse = llm_setup()

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User prompt: {user_prompt}"},
        ],
        max_tokens=200,
        response_format=PromptCategoryResponse,
    )

    result = completion.choices[0].message.parsed
    if isinstance(result, PromptCategoryResponse):
        return cast(str, result.category.value)
    raise ValueError("Unexpected response format")


def categorize_user_messages(update_all_messages: bool) -> None:
    """Categorize user chat messages (prompts) using a zero-shot LLM.

    Args:
        update_all_messages (bool): If True, re-categorize all messages.
            Otherwise, only categorize messages without a Category.
    """
    with Session(get_engine()) as session:
        query = select(ChatMessage).where(ChatMessage.message_type == MessageType.USER_MESSAGE)
        if not update_all_messages:
            query = query.where(ChatMessage.category_id.is_(None))  # type: ignore

        categories = session.exec(select(Category).where(Category.name != OVERALL_CATEGORY_NAME)).all()
        category_dict = {category.name: category for category in categories}

        total_categorized = 0
        total_uncategorized = 0

        for i, message in enumerate(session.exec(query).all()):
            category_name = prompt_category_by_llm(message.content)
            category = category_dict.get(category_name)

            if category:
                message.category = category
                total_categorized += 1
            else:
                logger.warning(f"Category '{category_name}' not found for message {message.message_id}")
                total_uncategorized += 1

            # Chunk to prevent commits from failing
            chunk_size = 100
            if (i + 1) % chunk_size == 0:
                session.commit()
                logger.info(f"Committed chunk of {chunk_size} messages")

        session.commit()
        logger.info("Committed final chunk of messages")

    logger.info(
        f"Prompt categorization complete. "
        f"Total messages categorized: {total_categorized}. "
        f"Total messages uncategorized: {total_uncategorized}."
    )
