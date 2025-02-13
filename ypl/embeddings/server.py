import logging
import os
from typing import cast

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.getenv("MODEL_NAME", "BAAI/bge-m3")
model = SentenceTransformer(model_name)
model = model.to(device)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Serving model {model_name} on {device}.")


class EmbedRequest(BaseModel):
    texts: list[str]
    model_name: str = model_name


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest) -> EmbedResponse:
    if request.model_name != model_name:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} was not found.")
    try:
        embeddings = cast(np.ndarray, model.encode(request.texts, device=device, convert_to_numpy=True)).tolist()
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/status")
async def status() -> dict:
    if model:
        return {"status": "OK"}
    else:
        raise HTTPException(status_code=503, detail="Model not ready")
