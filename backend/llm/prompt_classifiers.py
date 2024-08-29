from enum import Enum
from functools import cache
from typing import Any, cast

from openai import OpenAI
from pydantic import BaseModel
from sqlmodel import select
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from backend import prompts
from backend.db import (
    Session,
    get_engine,
)
from db.ratings import (
    OVERALL_CATEGORY_NAME,
    Category,
)

client: OpenAI | None = None


def initialize_client(**kwargs: Any) -> OpenAI:
    global client
    if client is None:
        client = OpenAI(**kwargs)
    return client


def fetch_categories_with_descriptions_from_db() -> dict[str, str]:
    with Session(get_engine()) as session:
        categories = session.exec(
            select(Category.name, Category.description).where(Category.name != OVERALL_CATEGORY_NAME)
        ).all()
        return {name: description for name, description in categories if description is not None}


def construct_system_prompt(system_prompt: str, category_descriptions_dict: dict[str, str]) -> str:
    formatted_categories = [f"- {name}: {description}" for name, description in category_descriptions_dict.items()]
    return system_prompt.replace("<REPLACE_WITH_CATEGORIES_FROM_DB>", "\n".join(formatted_categories))


class PromptCategoryResponse(BaseModel):
    category: Enum


@cache
def llm_setup() -> tuple[str, type[PromptCategoryResponse]]:
    category_descriptions_dict = fetch_categories_with_descriptions_from_db()
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
