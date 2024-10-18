from functools import cache

import torch
from fastapi import APIRouter
from pydantic import BaseModel

from ypl.backend.config import settings
from ypl.pytorch.model.categorizer import CategorizerClassificationModel

router = APIRouter()


@cache
def get_categorizer_model() -> CategorizerClassificationModel:
    with torch.amp.autocast("cuda", dtype=torch.float16):
        categorizer_model = CategorizerClassificationModel.from_gcp_zip(settings.CATEGORIZER_MODEL_PATH)
        categorizer_model.cuda()
        categorizer_model.eval()
        categorizer_model.compile_cuda_graphs()

    return categorizer_model


get_categorizer_model()  # warm up the model


class CategorizeResponse(BaseModel):
    category: str | list[str]
    difficulty: int


class CategorizeRequest(BaseModel):
    prompt: str


@router.post("/categorize")
async def categorize(request: CategorizeRequest) -> CategorizeResponse:
    category, difficulty = await get_categorizer_model().acategorize(request.prompt)
    return CategorizeResponse(category=category, difficulty=difficulty)
