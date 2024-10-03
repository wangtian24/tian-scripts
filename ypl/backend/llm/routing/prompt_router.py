import ast
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from ypl.backend.llm.chat import ModelInfo, get_chat_model
from ypl.backend.llm.constants import MODEL_DESCRIPTIONS
from ypl.backend.llm.labeler import LLMLabeler
from ypl.backend.llm.routing.policy import SelectionCriteria
from ypl.backend.llm.routing.router import ModelProposer, RouterState
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


class ZeroShotPromptQualityProposer(ModelProposer):
    def __init__(self, model_info: ModelInfo, prompt: str):
        self.chat_model = get_chat_model(model_info)
        self.prompt = prompt

    def _propose_models(self, num_models: int, models_to_select: set[str], state: RouterState) -> RouterState:
        model_metadata = {model: MODEL_DESCRIPTIONS[model] for model in models_to_select if model in MODEL_DESCRIPTIONS}
        labeler = ZeroShotPromptQualityLabeler(model_metadata, self.chat_model)
        models = [x for x in labeler.label(self.prompt) if x in models_to_select][:num_models]

        return RouterState(
            selected_models={
                model: {SelectionCriteria.BEST_PROMPT_QUALITY: (1 / (r + 1))} for r, model in enumerate(models)
            },
            all_models=models_to_select,
        )

    async def _apropose_models(self, num_models: int, models_to_select: set[str], state: RouterState) -> RouterState:
        model_metadata = {model: MODEL_DESCRIPTIONS[model] for model in models_to_select if model in MODEL_DESCRIPTIONS}
        labeler = ZeroShotPromptQualityLabeler(model_metadata, self.chat_model)
        models = [x for x in await labeler.alabel(self.prompt) if x in models_to_select][:num_models]

        return RouterState(
            selected_models={
                model: {SelectionCriteria.BEST_PROMPT_QUALITY: (1 / (r + 1))} for r, model in enumerate(models)
            },
            all_models=models_to_select,
        )
