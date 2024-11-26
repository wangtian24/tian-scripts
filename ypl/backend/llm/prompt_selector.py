from collections import defaultdict
from copy import deepcopy
from typing import Self

from cachetools.func import ttl_cache
from sqlalchemy import select
from sqlmodel import Session

from ypl.backend.db import get_engine
from ypl.db.chats import PromptModifier
from ypl.utils import RNGMixin


class PromptModifierSelector:
    """
    Selects a set of prompt modifiers for a given list of models.
    """

    def select_modifiers(self, models: list[str]) -> dict[str, tuple[str, str]]:
        """
        Selects a set of prompt modifiers for a given list of models.

        Args:
            models: A list of model names.

        Returns:
            A dictionary mapping each model name to (ID, prompt modifier).
        """
        raise NotImplementedError


class CategorizedPromptModifierSelector(RNGMixin, PromptModifierSelector):
    """
    Selects a set of prompt modifiers from different categories.
    """

    def __init__(
        self,
        system_modifiers: list[tuple[str, str, str]],
        model_modifier_history: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        """
        Args:
            system_prompts: A list of tuples of (category, prompt modifier ID, system prompt modifier).
            model_modifier_history: A dictionary mapping model names to the chosen prompt modifier.
        """
        sys_modifiers_dict: dict[str, set[tuple[str, str]]] = defaultdict(set)

        for category, modifier_id, modifier in system_modifiers:
            sys_modifiers_dict[category].add((modifier_id, modifier))

        if not sys_modifiers_dict:
            sys_modifiers_dict["default"] = {("", "")}

        self.model_modifier_history = model_modifier_history or {}
        self.system_modifiers = dict(sys_modifiers_dict)

    def select_modifiers(self, models: list[str]) -> dict[str, tuple[str, str]]:
        system_modifiers = deepcopy(self.system_modifiers)
        model_mod_map = {}

        for model in models:
            if not system_modifiers:
                system_modifiers = deepcopy(self.system_modifiers)  # repeat the process

            try:
                model_mod_map[model] = self.model_modifier_history[model]
                continue
            except KeyError:
                pass

            cat_idx = self.get_rng().randint(0, len(system_modifiers.keys()))
            cat = list(system_modifiers.keys())[cat_idx]
            modifier_idx = self.get_rng().randint(0, len(system_modifiers[cat]))
            modifier = list(system_modifiers[cat])[modifier_idx]
            model_mod_map[model] = modifier
            system_modifiers[cat].remove(modifier)

            if not system_modifiers[cat]:
                del system_modifiers[cat]

        return model_mod_map

    @classmethod
    def make_default(cls) -> Self:
        return cls(
            [
                ("length", "", "Limit the response to 50 words or fewer."),
                ("length", "", "Provide a detailed explanation, with step-by-step instructions if applicable."),
                ("tone", "", "Use a conversational, friendly tone."),
                ("tone", "", "Maintain a neutral and professional tone."),
                ("tone", "", "Use a humorous and light-hearted tone."),
                ("complexity", "", "Respond as if talking to a high-school student."),
                ("complexity", "", "Respond as if talking to a college graduate."),
                ("complexity", "", "Respond as if talking to a 5-year-old child."),
                ("formatting", "", "Use bullet points and numbered lists."),
                ("formatting", "", "Use markdown with bullet points in hierarchical form."),
                ("formatting", "", "Write in highly structured prose."),
            ]
        )

    @classmethod
    @ttl_cache(ttl=600)  # 10 minutes
    def make_default_from_db(cls) -> Self:
        query = select(PromptModifier)

        with Session(get_engine()) as session:
            modifiers = session.exec(query).scalars().all()  # type: ignore[call-overload]

        return cls(
            [
                (modifier.category, str(modifier.prompt_modifier_id), modifier.text)
                for modifier in modifiers
                if modifier.text is not None
            ]
        )
