from typing import Any

from fastapi import APIRouter, Query

from backend.llm.chat import (
    DEFAULT_HIGH_SIM_THRESHOLD,
    DEFAULT_UNIQUENESS_THRESHOLD,
    highlight_llm_similarities_with_embeddings,
)

router = APIRouter()


@router.post("/highlight_similar_content")
def highlight_similar_content(
    response_a: str = Query(..., description="First response to compare"),
    response_b: str = Query(..., description="Second response to compare"),
    high_sim_threshold: float = Query(DEFAULT_HIGH_SIM_THRESHOLD, description="High similarity threshold"),
    uniqueness_threshold: float = Query(DEFAULT_UNIQUENESS_THRESHOLD, description="Uniqueness threshold"),
) -> dict[str, Any]:
    return highlight_llm_similarities_with_embeddings(
        response_a,
        response_b,
        high_sim_threshold,
        uniqueness_threshold,
    )
