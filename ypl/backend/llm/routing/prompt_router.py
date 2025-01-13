import ast
import logging
from typing import Any

import aiohttp
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from ypl.backend.llm.constants import MODEL_DESCRIPTIONS, MODEL_HEURISTICS
from ypl.backend.llm.db_helpers import get_chat_model
from ypl.backend.llm.labeler import LLMLabeler
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.prompt_classifiers import RemotePromptCategorizer
from ypl.backend.llm.routing.modules.proposers import ModelProposer
from ypl.backend.llm.routing.policy import SelectionCriteria
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.prompts import PROMPTS_MODEL_QUALITY_PROMPT_TEMPLATE


class ZeroShotPromptQualityLabeler(LLMLabeler[str, list[str]]):
    def __init__(self, model_metadata: dict[str, str], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_metadata = model_metadata

    @property
    def error_value(self) -> list[str]:
        return []

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return PROMPTS_MODEL_QUALITY_PROMPT_TEMPLATE | llm  # type: ignore

    def _prepare_input(self, prompt: str) -> dict[str, Any]:
        return {"prompt": prompt, "model_metadata": self.model_metadata}

    def _parse_output(self, output: BaseMessage) -> list[str]:
        if isinstance(output.content, str):
            ret = ast.literal_eval(output.content)
        else:
            return self.error_value

        if not isinstance(ret, list):
            return self.error_value

        return ret


class RemotePromptCategorizerProposer(ModelProposer):
    def __init__(
        self,
        prompt: str,
        api_endpoint: str,
        api_key: str,
        skill_deficit_threshold: int = 0,
        exclude_unknown_models: bool = True,
    ) -> None:
        """
        Args:
            prompt: The prompt to categorize.
            api_endpoint: The endpoint of the remote categorizer.
            api_key: The API key to use for the remote categorizer.
            skill_deficit_threshold: The skill deficit threshold. Models with `difficulty - skill` less than
                this threshold are excluded.
            exclude_unknown_models: Whether to exclude unknown models. If false, mistral-large-latest is assumed to be
                representative of all unknown models in quality.
        """
        self.prompt = prompt
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.skill_deficit_threshold = -skill_deficit_threshold
        self.exclude_unknown_models = exclude_unknown_models

    def _select_models_from_category(
        self, response: dict[str, Any], models_to_select: set[str]
    ) -> tuple[list[tuple[float, str]], set[str]]:
        """
        Helper function to select models from a category response.

        Returns:
            A tuple containing the selected models and the excluded models.
        """
        category, difficulty = response["category"], response["difficulty"]
        models = []
        excluded_models = set()

        logging.info(f"Prompt categorizer response: {response}")

        for model in models_to_select:
            if model not in MODEL_HEURISTICS and self.exclude_unknown_models:
                excluded_models.add(model)
                continue

            heuristics = MODEL_HEURISTICS[model]  # defaultdict; see constants.py for the default heuristics
            quality = heuristics.estimate_quality(category, difficulty)

            if quality < self.skill_deficit_threshold:
                excluded_models.add(model)
                continue

            models.append((quality, model))

        return sorted(models, key=lambda x: x[0], reverse=True), excluded_models

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        categorizer = RemotePromptCategorizer(self.api_endpoint, self.api_key)
        response = categorizer.categorize(self.prompt)

        return self._propose_models_from_category(dict(category=response.category), models_to_select)

    def _propose_models_from_category(self, response: dict[str, Any], models_to_select: set[str]) -> RouterState:
        selected_models, excluded_models = self._select_models_from_category(response, models_to_select)

        return RouterState(
            selected_models={
                model: {SelectionCriteria.PROMPT_CATEGORIZER: quality} for quality, model in selected_models
            },
            all_models=models_to_select,
            excluded_models=excluded_models,
        )

    async def _apropose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_endpoint + "/categorize",
                json={"prompt": self.prompt},
                headers={"X-API-KEY": self.api_key},
            ) as response:
                json_response = await response.json()

        return self._propose_models_from_category(json_response, models_to_select)


class ZeroShotPromptQualityProposer(ModelProposer):
    def __init__(self, model_info: ModelInfo, prompt: str):
        self.chat_model = get_chat_model(model_info)
        self.prompt = prompt

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        model_metadata = {model: MODEL_DESCRIPTIONS[model] for model in models_to_select if model in MODEL_DESCRIPTIONS}
        labeler = ZeroShotPromptQualityLabeler(model_metadata, self.chat_model)
        models = [x for x in labeler.label(self.prompt) if x in models_to_select]

        return RouterState(
            selected_models={
                model: {SelectionCriteria.BEST_PROMPT_QUALITY: (1 / (r + 1))} for r, model in enumerate(models)
            },
            all_models=models_to_select,
        )

    async def _apropose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        model_metadata = {model: MODEL_DESCRIPTIONS[model] for model in models_to_select if model in MODEL_DESCRIPTIONS}
        labeler = ZeroShotPromptQualityLabeler(model_metadata, self.chat_model)
        models = [x for x in await labeler.alabel(self.prompt) if x in models_to_select]

        return RouterState(
            selected_models={
                model: {SelectionCriteria.BEST_PROMPT_QUALITY: (1 / (r + 1))} for r, model in enumerate(models)
            },
            all_models=models_to_select,
        )
