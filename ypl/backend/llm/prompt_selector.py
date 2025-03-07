import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Self

from cachetools.func import ttl_cache
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import Insert as pg_insert
from sqlmodel import Session

from ypl.backend.db import get_async_session, get_engine
from ypl.db.chats import (
    ChatMessage,
    MessageModifierStatus,
    MessageType,
    ModifierCategory,
    PromptModifier,
    PromptModifierAssoc,
    Turn,
)
from ypl.db.language_models import LanguageModel
from ypl.utils import RNGMixin


@dataclass
class PromptModifierPolicy:
    # How many categories are modified per model.
    num_categories_to_modify: int
    # Whether to use the same modified categories for all models.
    same_categories_all_models: bool
    # Whether to modify all models or just one.
    modify_all_models: bool
    # Whether to reuse existing modifiers for a model.
    reuse_previous_modifiers: bool
    # Whether to modify the last unmodified model or a random one.
    modify_last_model: bool


DEFAULT_PROMPT_MODIFIER_POLICY = PromptModifierPolicy(
    num_categories_to_modify=1,
    same_categories_all_models=True,
    modify_all_models=False,
    reuse_previous_modifiers=True,
    modify_last_model=True,
)


async def get_modifiers_by_model_and_position(
    chat_id: str,
) -> tuple[dict[str, list[str]], tuple[str | None, str | None]]:
    """
    Fetches the modifier history for a given chat from the DB.
    Returns a tuple of:
    - a mapping between the model name and a list of modifier IDs.
    - a tuple of the modifier most recently applied to the LHS and the one most recently applied to the RHS.
    """
    async with get_async_session() as session:
        query = (
            select(LanguageModel.internal_name, PromptModifierAssoc.prompt_modifier_id)  # type: ignore
            .join(ChatMessage, ChatMessage.assistant_language_model_id == LanguageModel.language_model_id)
            .join(Turn, ChatMessage.turn_id == Turn.turn_id)
            .outerjoin(PromptModifierAssoc, PromptModifierAssoc.chat_message_id == ChatMessage.message_id)
            .where(Turn.chat_id == uuid.UUID(chat_id))
        )
        modifiers_by_model = defaultdict(list)
        for model_name, modifier_id in (await session.exec(query)).all():
            if modifier_id is not None:
                modifiers_by_model[model_name].append(str(modifier_id))

        modifiers_by_position = (None, None)
        # Get the 2 most recent messages with modifiers
        recent_query = (
            select(ChatMessage.message_id, PromptModifierAssoc.prompt_modifier_id)  # type: ignore
            .join(Turn, ChatMessage.turn_id == Turn.turn_id)
            .outerjoin(PromptModifierAssoc, PromptModifierAssoc.chat_message_id == ChatMessage.message_id)
            .where(Turn.chat_id == uuid.UUID(chat_id), ChatMessage.modifier_status == MessageModifierStatus.SELECTED)
            .order_by(Turn.sequence_id.desc(), ChatMessage.turn_sequence_number.desc())  # type: ignore
            .limit(2)
        )
        res = (await session.exec(recent_query)).all()
        if res and len(res) == 2:
            # Reverse the order of the results to get the LHS and RHS modifiers.
            modifiers_by_position = (res[1][1], res[0][1])

        return dict(modifiers_by_model), modifiers_by_position


async def store_modifiers(turn_id: str, modifiers: dict[str, list[tuple[str, str]]]) -> None:
    """Stores the prompt modifiers for a given turn in the DB."""
    async with get_async_session() as session:
        query = (
            select(ChatMessage.message_id, LanguageModel.internal_name)  # type: ignore
            .join(LanguageModel, ChatMessage.assistant_language_model_id == LanguageModel.language_model_id)
            .where(
                ChatMessage.turn_id == turn_id,
                ChatMessage.message_type == MessageType.ASSISTANT_MESSAGE,
                ChatMessage.deleted_at.is_(None),  # type: ignore
            )
        )
        results = await session.exec(query)
        message_models = {str(message_id): model_name for message_id, model_name in results}

        assocs = []
        for message_id, model_name in message_models.items():
            if model_name in modifiers:
                for modifier_id, _ in modifiers[model_name]:
                    assoc = {"chat_message_id": message_id, "prompt_modifier_id": modifier_id}
                    assocs.append(assoc)

        if assocs:
            stmt = pg_insert(PromptModifierAssoc).values(assocs)
            stmt = stmt.on_conflict_do_nothing()
            await session.exec(stmt)  # type: ignore
            await session.commit()


class CategorizedPromptModifierSelector(RNGMixin):
    """
    Selects a set of prompt modifiers from different categories.
    """

    def __init__(
        self,
        system_modifiers: list[PromptModifier],
        policy: PromptModifierPolicy = DEFAULT_PROMPT_MODIFIER_POLICY,
    ) -> None:
        """
        Args:
            system_modifiers: The modifiers available to select from.
            policy: The policy to use when selecting modifiers.
        """
        if not system_modifiers:
            raise ValueError("No system modifiers provided.")

        self.modifiers_by_category: dict[str, list[PromptModifier]] = defaultdict(list)
        self.modifiers_by_id: dict[str, PromptModifier] = {}
        for modifier in system_modifiers:
            self.modifiers_by_category[modifier.category.value].append(modifier)
            self.modifiers_by_id[str(modifier.prompt_modifier_id)] = modifier

        self.policy = policy

    def select_modifiers(
        self,
        models: list[str],
        modifier_history: dict[str, list[str]] | None = None,
        applicable_modifiers: list[str] | None = None,
        modifiers_by_position: tuple[str | None, str | None] = (None, None),
        should_auto_select_modifiers: bool = True,
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Selects a set of prompt modifiers for a given list of models.

        If a model has already been modified, the same modifiers will be applied to it again.
        Other models will be modified according to the selection policy.

        Args:
            models: A list of model names.
            modifier_history: A dictionary mapping model names to previously chosen modifier IDs.
            applicable_modifiers: A list of modifier IDs that are applicable to the models.
            modifiers_by_position: A tuple of the modifier most recently applied to the LHS and RHS.
            should_auto_select_modifiers: Whether to select modifiers for unmodified models.

        Returns:
            A dictionary mapping each model name to its selected modifier, as a tuple of (ID, text).
        """
        if modifier_history is None:
            modifier_history = {}
        if applicable_modifiers is None:
            applicable_modifiers = []

        modifiers_by_model: dict[str, list[PromptModifier]] = {}
        lhs_modifier, rhs_modifier = None, None

        # First, apply any modifiers that were most recently applied to the LHS and RHS.
        if len(modifiers_by_position) == len(models) == 2:
            lhs_modifier, rhs_modifier = modifiers_by_position
            if lhs_modifier:
                modifiers_by_model[models[0]] = [self.modifiers_by_id[str(lhs_modifier)]]
                logging.debug(f"Applying LHS modifier {lhs_modifier} to model {models[0]}")
            if rhs_modifier:
                modifiers_by_model[models[-1]] = [self.modifiers_by_id[str(rhs_modifier)]]
                logging.debug(f"Applying RHS modifier {rhs_modifier} to model {models[-1]}")

        # Next, apply any modifiers that were previously selected.
        if self.policy.reuse_previous_modifiers:
            for model in models:
                if model in modifiers_by_model:
                    continue
                if modifier_history.get(model):
                    modifiers_by_model[model] = [
                        self.modifiers_by_id[id] for id in modifier_history[model] if id in self.modifiers_by_id
                    ]
                    logging.debug(f"Reusing modifier {modifiers_by_model[model]} for model {model}")
        unmodified_models = [model for model in models if model not in modifiers_by_model]

        if (
            not unmodified_models  # No models to modify.
            or not should_auto_select_modifiers  # Client requested no modifiers.
            or lhs_modifier  # Modifiers were already applied due to LHS/RHS selection.
            or rhs_modifier  # Modifiers were already applied due to LHS/RHS selection.
        ):
            return self._to_modifier_history(modifiers_by_model)

        # Select which models to modify.
        if self.policy.modify_all_models:
            models_to_modify = unmodified_models
        else:
            if self.policy.modify_last_model:
                # Modify just the last unmodified model.
                models_to_modify = [unmodified_models[-1]]
            else:
                # Modify a random unmodified model.
                models_to_modify = [self.get_rng().choice(unmodified_models)]

        # Select the modifier categories and modifiers to apply to each model.
        categories_to_modify = []
        if self.policy.same_categories_all_models:
            categories_to_modify_set = set()
            # Choose the categories used by already modified models.
            for modifiers in modifiers_by_model.values():
                categories_to_modify_set.update({modifier.category.value for modifier in modifiers})
            categories_to_modify = list(categories_to_modify_set)
        # If too few, add new ones.
        if len(categories_to_modify) < self.policy.num_categories_to_modify:
            unused_categories = list(set(self.modifiers_by_category.keys()) - set(categories_to_modify))
            num_new_categories = self.policy.num_categories_to_modify - len(categories_to_modify)
            categories_to_modify.extend(self.get_rng().choice(unused_categories, num_new_categories, replace=False))

        if not categories_to_modify:
            return self._to_modifier_history(modifiers_by_model)

        # Select the modifiers to apply to each model.
        already_used_modifiers = set()
        for model in models_to_modify:
            modifiers = []
            for category in categories_to_modify:
                available_modifier_ids: list[str] = [
                    str(modifier.prompt_modifier_id)
                    for modifier in self.modifiers_by_category[category]
                    if str(modifier.prompt_modifier_id) not in already_used_modifiers
                    and modifier.name in applicable_modifiers
                ]
                if not available_modifier_ids:
                    continue
                modifier_id = self.get_rng().choice(available_modifier_ids)
                modifiers.append(self.modifiers_by_id[modifier_id])
                already_used_modifiers.add(modifier_id)
            modifiers_by_model[model] = modifiers
            logging.debug(f"Applying modifier {modifiers} to model {model}")

        # Return just the modifier IDs and texts.
        return self._to_modifier_history(modifiers_by_model)

    @classmethod
    def _to_modifier_history(
        cls, modifiers_by_model: dict[str, list[PromptModifier]]
    ) -> dict[str, list[tuple[str, str]]]:
        return {
            model: list(set([(str(modifier.prompt_modifier_id), modifier.text) for modifier in modifiers]))
            for model, modifiers in modifiers_by_model.items()
        }

    @classmethod
    def make_default(cls, **kwargs: Any) -> Self:
        return cls(
            system_modifiers=[
                PromptModifier(category=cat, name=name, text=text, id=f"id-{i}")
                for i, (cat, name, text) in enumerate(
                    [
                        (ModifierCategory.length, "length_under_50", "Limit the response to 50 words or fewer."),
                        (
                            ModifierCategory.length,
                            "length_long",
                            "Provide a detailed response; step-by-step instructions if needed.",
                        ),
                        (ModifierCategory.tone, "tone_conversational", "Use a conversational, friendly tone."),
                        (ModifierCategory.tone, "tone_neutral", "Maintain a neutral and professional tone."),
                        (ModifierCategory.tone, "tone_humorous", "Use a humorous and light-hearted tone."),
                        (
                            ModifierCategory.complexity,
                            "complexity_high_school",
                            "Respond as if talking to a high school student.",
                        ),
                        (
                            ModifierCategory.complexity,
                            "complexity_college",
                            "Respond as if talking to a college graduate.",
                        ),
                        (
                            ModifierCategory.complexity,
                            "complexity_child",
                            "Respond as if talking to a 5-year-old child.",
                        ),
                        (
                            ModifierCategory.formatting,
                            "formatting_bullet_points",
                            "Use bullet points and numbered lists.",
                        ),
                        (
                            ModifierCategory.formatting,
                            "formatting_markdown",
                            "Use markdown with bullet points in hierarchical form.",
                        ),
                        (ModifierCategory.formatting, "formatting_structured", "Write in highly structured prose."),
                    ]
                )
            ],
            **kwargs,
        )

    @classmethod
    @ttl_cache(ttl=600)  # 10 minutes
    def make_default_from_db(cls, **kwargs: Any) -> Self:
        with Session(get_engine()) as session:
            return cls(
                system_modifiers=list(
                    session.scalars(select(PromptModifier).where(PromptModifier.deleted_at.is_(None))).all()  # type: ignore
                ),
                **kwargs,
            )
