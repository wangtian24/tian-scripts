import logging
from functools import cache

from fastapi import APIRouter
from pydantic import BaseModel

from ypl.backend.config import settings
from ypl.pytorch.model.categorizer import OnlinePromptClassifierModel

router = APIRouter()
logger = logging.getLogger()


@cache
def get_categorizer_model() -> OnlinePromptClassifierModel:
    logger.info(settings.CATEGORIZER_MODEL_PATH)

    categorizer_model = OnlinePromptClassifierModel.from_gcp_zip(settings.CATEGORIZER_MODEL_PATH)
    categorizer_model.cuda()
    categorizer_model.eval()
    categorizer_model.half()
    categorizer_model.compile_cuda_graphs()

    return categorizer_model


get_categorizer_model()  # warm up the model


class CategorizeResponse(BaseModel):
    category: str | list[str]


class CategorizeRequest(BaseModel):
    prompt: str


@router.post("/categorize")
async def categorize(request: CategorizeRequest) -> CategorizeResponse:
    category = await get_categorizer_model().acategorize(request.prompt)

    return CategorizeResponse(category=category)
