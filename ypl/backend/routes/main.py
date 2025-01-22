import asyncio
import logging
import os

import nltk
import sqlalchemy as sa
from dotenv import load_dotenv
from fastapi import APIRouter, Depends

from ypl.backend.config import settings
from ypl.backend.llm.provider.provider_clients import load_active_models_with_providers
from ypl.backend.llm.ranking import get_ranker
from ypl.backend.prompts import load_prompt_modifiers
from ypl.backend.routes.api_auth import validate_api_key
from ypl.backend.routes.v1 import chat_completions, chat_feed, credit, health, model, payment, rank, reward
from ypl.backend.routes.v1 import chats as chats_route
from ypl.backend.routes.v1 import email as email_route
from ypl.backend.routes.v1 import event as event_route
from ypl.backend.routes.v1 import files as files_route
from ypl.backend.routes.v1 import points as points_route
from ypl.backend.routes.v1 import profile as profile_route
from ypl.backend.routes.v1 import provider as provider_route
from ypl.backend.routes.v1 import route as llm_route
from ypl.backend.routes.v1 import user as user_route
from ypl.backend.routes.v1 import user_capability as user_capability_route
from ypl.backend.utils.json import json_dumps


def log_sql_query(conn, cursor, statement, parameters, context, executemany):  # type: ignore
    log_dict = {
        "message": "SQL Query",
        "statement": statement,
    }
    if parameters:
        log_dict["parameters"] = parameters

    logging.info(json_dumps(log_dict))


def app_init() -> None:
    load_dotenv()
    nltk_data_path = os.getenv("NLTK_DATA")
    if nltk_data_path:
        nltk.data.path.append(nltk_data_path)
    if settings.ENVIRONMENT != "production":
        sa.event.listen(sa.engine.Engine, "before_cursor_execute", log_sql_query)

    # prefetches the models-providers info & caches the data
    load_active_models_with_providers()
    # prefetches the prompt-modifiers & caches the data
    load_prompt_modifiers()

    asyncio.create_task(get_ranker().add_evals_from_db())


api_router = APIRouter(dependencies=[Depends(validate_api_key)])
api_router.include_router(credit.router, prefix="/v1", tags=["credit"])
api_router.include_router(health.router, prefix="/v1", tags=["health"])
api_router.include_router(llm_route.router, prefix="/v1", tags=["route"])
api_router.include_router(rank.router, prefix="/v1", tags=["leaderboard"])
api_router.include_router(reward.router, prefix="/v1", tags=["reward"])
api_router.include_router(model.router, prefix="/v1", tags=["models"])
api_router.include_router(payment.router, prefix="/v1", tags=["payment"])
api_router.include_router(provider_route.router, prefix="/v1", tags=["providers"])
api_router.include_router(chat_feed.router, prefix="/v1", tags=["chat-feed"])
api_router.include_router(chats_route.router, prefix="/v1", tags=["chats"])
api_router.include_router(chat_completions.router, prefix="/v1", tags=["chat_completions"])
api_router.include_router(files_route.router, prefix="/v1", tags=["files"])
api_router.include_router(email_route.router, prefix="/v1", tags=["emails"])
api_router.include_router(user_route.router, prefix="/v1", tags=["user"])
api_router.include_router(points_route.router, prefix="/v1", tags=["admin"])
api_router.include_router(user_capability_route.router, prefix="/v1", tags=["admin"])
api_router.include_router(event_route.router, prefix="/v1", tags=["admin"])
api_router.include_router(profile_route.router, prefix="/v1", tags=["profile"])
app_init()
