import pytest

from ypl.backend.llm.conversational.butler import (
    Butler,
    ButlerIntentEnum,
    ButlerRequest,
    ButlerResponse,
    ChosenModel,
    DeepIntent,
    InteractionType,
    IntroduceResponsesRequest,
    ScratchcardAction,
    UserActionRequest,
)
from ypl.backend.llm.prompt_classifiers import CategorizerResponse, PromptCategorizer


class DummyPromptCategorizer(PromptCategorizer):
    async def acategorize(self, prompt: str) -> CategorizerResponse:
        return CategorizerResponse(category="test")


def test_deep_intent_add() -> None:
    assert DeepIntent(span=(0, 1), intent=ButlerIntentEnum.CHAT_NEW) + 1 == DeepIntent(
        span=(1, 2), intent=ButlerIntentEnum.CHAT_NEW
    )
    assert 1 + DeepIntent(span=(0, 1), intent=ButlerIntentEnum.CHAT_NEW) == DeepIntent(
        span=(1, 2), intent=ButlerIntentEnum.CHAT_NEW
    )


def test_butler_response_make() -> None:
    assert ButlerResponse.make("Hello, world!") == ButlerResponse(response="Hello, world!", intents=[])
    assert (
        ButlerResponse.make("123456", intent=ButlerIntentEnum.RESPONSE_NEW)
        + ButlerResponse.make("123456", intent=ButlerIntentEnum.CHAT_NEW)
    ) == ButlerResponse(
        response="123456123456",
        intents=[
            DeepIntent(span=(0, 6), intent=ButlerIntentEnum.RESPONSE_NEW),
            DeepIntent(span=(6, 12), intent=ButlerIntentEnum.CHAT_NEW),
        ],
    )


@pytest.mark.asyncio
async def test_butler_fsm() -> None:
    butler = Butler(categorizer=DummyPromptCategorizer())
    response = await butler.acall(
        ButlerRequest(
            interaction_type=InteractionType.INVITATION,
        )
    )
    assert response == ButlerResponse(response="What's on your mind?", intents=[])

    response = await butler.acall(
        IntroduceResponsesRequest(
            interaction_type=InteractionType.PRE_CHAT_COMPLETION,
            user_prompt="Hello, world!",
            models=[
                ChosenModel(name="gpt-1", source="user"),
                ChosenModel(name="gpt-2", source="system"),
            ],
        )
    )
    assert response == ButlerResponse(
        response=(
            "On the left is the AI you requested. On the right is another I chose to help with test. "
            "Let me know which you like better."
        ),
        intents=[],
    )

    response = await butler.acall(
        UserActionRequest(
            interaction_type=InteractionType.SCRATCHCARD_ACTION,
            action=ScratchcardAction.NO_SCRATCHCARD,
            preferred_model=ChosenModel(name="gpt-1", source="user"),
        )
    )
    assert response == ButlerResponse(
        response="You chose gpt-1. Continue with it or start a new topic.",
        intents=[
            DeepIntent(span=(37, 54), intent=ButlerIntentEnum.CHAT_NEW),
        ],
    )


@pytest.mark.asyncio
async def test_butler_serialization() -> None:
    butler = Butler(categorizer=DummyPromptCategorizer())
    response = await butler.acall(
        ButlerRequest(
            interaction_type=InteractionType.INVITATION,
        )
    )
    assert response == ButlerResponse(response="What's on your mind?", intents=[])

    state_dict = butler.state_dict()
    butler = Butler(categorizer=DummyPromptCategorizer())
    butler.load_state_dict(state_dict)

    response = await butler.acall(
        IntroduceResponsesRequest(
            interaction_type=InteractionType.PRE_CHAT_COMPLETION,
            user_prompt="Hello, world!",
            models=[
                ChosenModel(name="gpt-1", source="user"),
                ChosenModel(name="gpt-2", source="system"),
            ],
        )
    )
    assert response == ButlerResponse(
        response=(
            "On the left is the AI you requested. On the right is another I chose to help with test. "
            "Let me know which you like better."
        ),
        intents=[],
    )
