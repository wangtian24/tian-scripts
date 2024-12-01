from collections import defaultdict

from ypl.backend.llm.prompt_selector import CategorizedPromptModifierSelector, PromptModifierPolicy
from ypl.db.chats import PromptModifier


def test_default() -> None:
    selector = CategorizedPromptModifierSelector.make_default()
    assert len(selector.modifiers_by_category) == 4  # categories
    assert sum(len(modifiers) for modifiers in selector.modifiers_by_category.values()) == len(selector.modifiers_by_id)


def test_select_modifiers() -> None:
    selector = CategorizedPromptModifierSelector.make_default(
        policy=PromptModifierPolicy(
            num_categories_to_modify=1,
            same_categories_all_models=True,
            modify_all_models=True,
            reuse_previous_modifiers=True,
        )
    )
    selector.set_seed(123, overwrite_existing=True)
    tone_modifier1, tone_modifier2 = selector.modifiers_by_category["tone"][:2]
    length_modifier = selector.modifiers_by_category["length"][0]

    models = ["model1", "model2", "model3", "model4"]
    history: dict[str, list[str]] = {
        "model1": [str(tone_modifier1.prompt_modifier_id), str(length_modifier.prompt_modifier_id)],
        "model2": [str(tone_modifier2.prompt_modifier_id)],
        "model3": [],
    }

    modifiers = selector.select_modifiers(models, history)

    # All models should be modified.
    assert len(modifiers) == len(models)

    for model in ["model3", "model4"]:
        assert len(modifiers[model]) > 0
        # Modifiers should be from one of the categories that was modified in the other models.
        for modifier_id, _ in modifiers[model]:
            modifier_category = selector.modifiers_by_id[modifier_id].category.value
            assert modifier_category in {"length", "tone"}

    # If we don't modify all models, only one should be modified.
    selector.policy.modify_all_models = False
    modifiers = selector.select_modifiers(models, history)

    assert len(modifiers) < len(models)

    # But history should still be respected.
    for model in ["model1", "model2"]:
        assert {modifier_id for modifier_id, _ in modifiers[model]} == set(history[model])

    selector.policy.reuse_previous_modifiers = False
    selector.policy.modify_all_models = True
    modifiers = selector.select_modifiers(models, history)
    assert len(modifiers) == len(models)
    # At least one of the new modifiers should be different from the modifier history.
    has_changed = False
    for model in models:
        if model in history:
            if {modifier_id for modifier_id, _ in modifiers[model]} != set(history[model]):
                has_changed = True
    assert has_changed

    selector.policy.modify_all_models = False
    randomly_modified_models = set()
    for _ in range(20):
        modifiers = selector.select_modifiers(models, history)
        # Every time, only one model should be modified.
        assert len(modifiers) == 1
        randomly_modified_models.add(list(modifiers.keys())[0])
    # But every model should be modified at least once.
    assert len(randomly_modified_models) == len(models)

    selector.policy.modify_all_models = True
    selector.policy.same_categories_all_models = True

    for num_categories_to_modify in range(1, 4):
        selector.policy.num_categories_to_modify = num_categories_to_modify
        modifiers = selector.select_modifiers(models)
        modifier_by_category: dict[str, list[PromptModifier]] = defaultdict(list)
        for model_modifiers in modifiers.values():
            for id, _ in model_modifiers:
                modifier = selector.modifiers_by_id[id]
                modifier_by_category[modifier.category.value].append(modifier)
        # Check that the correct number of categories were used.
        assert len(modifier_by_category) == num_categories_to_modify
        # Check that each modifier belongs to the correct category.
        for category, category_modifiers in modifier_by_category.items():
            assert all(modifier.category.value == category for modifier in category_modifiers)
