import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ypl.backend.llm.category_labeler import (
    get_prompt_category_classifier_llm,
    get_prompt_online_classifier_llm,
    merge_categories,
)
from ypl.backend.llm.judge import PromptModifierLabeler, YuppMultilabelClassifier, YuppOnlinePromptLabeler
from ypl.backend.llm.prompt_modifier import get_prompt_modifier_llm
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import StopWatch

router = APIRouter()

CLASSIFICATION_TIMEOUT_SECS: float = 1.25
CACHED_LABELERS: dict[str, Any] = {}


class PromptClassificationResponse(BaseModel):
    categories: list[str]
    modifiers: list[str]
    debug_info: dict[str, Any]


@router.post("/classify/prompt")
async def classify_prompt(prompt: str, model_name: str | None = None) -> PromptClassificationResponse:
    """
    An endpoint to classify a prompt into categories and modifiers. Can add other classifier in the future.
    This is mostly for test and evaluations.
    """
    try:
        # Get all labelers
        sw = StopWatch()
        model_key = model_name or "default"
        if model_key not in CACHED_LABELERS:
            online_labeler = YuppOnlinePromptLabeler(
                await get_prompt_online_classifier_llm(model_name), timeout_secs=CLASSIFICATION_TIMEOUT_SECS
            )
            sw.record_split("get_online_labeler")

            topic_labeler = YuppMultilabelClassifier(
                await get_prompt_category_classifier_llm(model_name), timeout_secs=CLASSIFICATION_TIMEOUT_SECS
            )
            sw.record_split("get_topic_labeler")

            modifier_labeler = PromptModifierLabeler(
                await get_prompt_modifier_llm(model_name), timeout_secs=CLASSIFICATION_TIMEOUT_SECS
            )
            sw.record_split("get_modifier_labeler")

            CACHED_LABELERS[model_key] = (online_labeler, topic_labeler, modifier_labeler)

        online_labeler, topic_labeler, modifier_labeler = CACHED_LABELERS[model_key]
        sw.record_split("get_modifier_labeler")

        online_label, topic_labels, prompt_modifier = await asyncio.gather(
            online_labeler.alabel(prompt),
            topic_labeler.alabel(prompt),
            modifier_labeler.alabel(prompt),
        )
        sw.end("labeling")
        prompt_categories = merge_categories(topic_labels=topic_labels, online_label=online_label)
        return PromptClassificationResponse(
            categories=prompt_categories, modifiers=prompt_modifier, debug_info={"latency_ms": sw.get_splits()}
        )

    except Exception as e:
        log_dict = {"message": f"Error classifying prompt: {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
