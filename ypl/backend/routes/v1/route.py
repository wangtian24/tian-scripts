from typing import Any

from fastapi import APIRouter, Body, Query
from tqdm import tqdm

from ypl.backend.llm.chat import SelectModelsV2Request, SelectModelsV2Response, select_models_plus
from ypl.backend.llm.ranking import get_ranker

router = APIRouter()


@router.post("/select_models_plus")
async def select_models_plus_route(request: SelectModelsV2Request) -> SelectModelsV2Response:
    return await select_models_plus(request)


@router.post("/update_ranker")
def update_ranker(
    model_a: str = Query(..., description="Model A"),  # noqa: B008
    model_b: str = Query(..., description="Model B"),  # noqa: B008
    result: float = Query(..., description="Outcome (for Model A)"),  # noqa: B008
) -> None:
    get_ranker().update(model_a, model_b, result)


@router.post("/bulk_update_ranker")
def bulk_update_ranker(
    updates: list[dict[str, Any]] = Body(  # noqa: B008
        ..., description="List of updates, each in the input format of `update_ranker`."
    ),
) -> None:
    for update in tqdm(updates, desc="Updating ranker", total=len(updates)):
        get_ranker().update(
            model_a=update["model_a"],
            model_b=update["model_b"],
            result_a=update["result"],
        )
