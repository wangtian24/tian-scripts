import os

import nltk
from dotenv import load_dotenv
from fastapi import APIRouter

from ypl.backend.llm.ranking import get_ranker
from ypl.backend.routes.v1 import health, highlight_similar_content, rank
from ypl.backend.routes.v1 import route as llm_route


def app_init() -> None:
    load_dotenv()
    nltk_data_path = os.getenv("NLTK_DATA")
    if nltk_data_path:
        nltk.data.path.append(nltk_data_path)
    get_ranker().add_evals_from_db()


api_router = APIRouter()
api_router.include_router(health.router, prefix="/v1", tags=["health"])
api_router.include_router(highlight_similar_content.router, prefix="/v1", tags=["highlight"])
api_router.include_router(llm_route.router, prefix="/v1", tags=["route"])
api_router.include_router(rank.router, prefix="/v1", tags=["models"])

app_init()
