from enum import Enum  # noqa: I001
from functools import cache
import logging
from typing import Any, cast

import aiohttp
from openai import OpenAI
from pydantic import BaseModel
import requests
from sqlmodel import select
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ypl.backend import prompts
from ypl.backend.db import Session, get_engine
from ypl.backend.llm.utils import fetch_categories_with_descriptions_from_db
from ypl.db.chats import ChatMessage, MessageType
from ypl.db.ratings import Category, OVERALL_CATEGORY_NAME
from ypl.backend.utils.json import json_dumps

client: OpenAI | None = None


class CategorizerResponse(BaseModel):
    category: str


class PromptCategorizer:
    async def acategorize(self, user_prompt: str) -> CategorizerResponse:
        return self.categorize(user_prompt)

    def categorize(self, user_prompt: str) -> CategorizerResponse:
        raise NotImplementedError


class RemotePromptCategorizer(PromptCategorizer):
    def __init__(self, api_endpoint: str, api_key: str) -> None:
        self.api_endpoint = api_endpoint
        self.api_key = api_key

    async def acategorize(self, user_prompt: str) -> CategorizerResponse:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_endpoint + "/categorize",
                json={"prompt": user_prompt},
                headers={"X-API-KEY": self.api_key},
            ) as response:
                json_response = await response.json()

        return CategorizerResponse(category=json_response["category"])

    def categorize(self, user_prompt: str) -> CategorizerResponse:
        return CategorizerResponse(
            category=requests.post(
                self.api_endpoint + "/categorize",
                json={"prompt": user_prompt},
                headers={"X-API-KEY": self.api_key},
            ).json()["category"]
        )


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
                log_dict = {
                    "message": "Category not found for message",
                    "category_name": category_name,
                    "message_id": str(message.message_id),
                }
                logging.warning(json_dumps(log_dict))
                total_uncategorized += 1

            # Chunk to prevent commits from failing
            chunk_size = 100
            if (i + 1) % chunk_size == 0:
                session.commit()
                log_dict = {
                    "message": f"Committed chunk of {chunk_size} messages",
                }
                logging.info(json_dumps(log_dict))

        session.commit()
        log_dict = {
            "message": "Committed final chunk of messages",
        }
        logging.info(json_dumps(log_dict))

    log_dict = {
        "message": "Prompt categorization complete.",
        "total_categorized": total_categorized,  # type: ignore
        "total_uncategorized": total_uncategorized,  # type: ignore
    }
    logging.info(json_dumps(log_dict))
