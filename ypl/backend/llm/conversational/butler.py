from enum import Enum
from typing import Any, Literal, Self

from pydantic import BaseModel

from ypl.backend.llm.conversational.base import StatefulAgent, register_state
from ypl.backend.llm.prompt_classifiers import PromptCategorizer


class ButlerIntentEnum(str, Enum):
    RESPONSE_NEW = "response.new"  # generate more responses
    CHAT_NEW = "chat.new"  # start a new chat


class DeepIntent(BaseModel):
    """
    A span of text in the user's message that is a specific intent, e.g., for marking a portion of the message as an
    action. An example is "Hey, want to buy a new phone?" with the span "buy a new phone" and the intent "buy.phone".
    For a list of butler intents, see :py:class:`.ButlerIntentEnum`.
    """

    span: tuple[int, int]
    intent: ButlerIntentEnum

    def __add__(self, offset: int) -> "DeepIntent":
        return DeepIntent(span=(self.span[0] + offset, self.span[1] + offset), intent=self.intent)

    def __iadd__(self, offset: int) -> "DeepIntent":
        self.span = (self.span[0] + offset, self.span[1] + offset)
        return self

    def __radd__(self, offset: int) -> "DeepIntent":
        return self.__add__(offset)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeepIntent):
            return False

        return self.span == other.span and self.intent == other.intent


class InteractionType(str, Enum):
    INVITATION = "invitation"
    PRE_CHAT_COMPLETION = "pre_chat_completion"
    CHAT_COMPLETION_ERROR = "chat_completion_error"
    SCRATCHCARD_ACTION = "scratchcard_action"


class ScratchcardAction(str, Enum):
    NO_SCRATCHCARD = "no_scratchcard"
    SCRATCHCARD = "scratchcard"
    AFTER_SCRATCHING = "after_scratching"


class ButlerRequest(BaseModel):
    interaction_type: InteractionType


class ChosenModel(BaseModel):
    name: str  # the internal name of the model
    source: Literal["user", "system"]  # user if the user chose the model, system if we chose it


class IntroduceResponsesRequest(ButlerRequest):
    models: list[ChosenModel]
    user_prompt: str

    def model_post_init(self, _: Any) -> None:
        assert self.interaction_type == InteractionType.PRE_CHAT_COMPLETION


class ErrorRequest(ButlerRequest):
    def model_post_init(self, _: Any) -> None:
        assert self.interaction_type == InteractionType.CHAT_COMPLETION_ERROR


class UserActionRequest(ButlerRequest):
    action: ScratchcardAction
    preferred_model: ChosenModel | None = None

    def model_post_init(self, _: Any) -> None:
        assert self.interaction_type == InteractionType.SCRATCHCARD_ACTION


class ButlerResponse(BaseModel):
    response: str
    intents: list[DeepIntent] = []

    def __add__(self, other: Self) -> "ButlerResponse":
        return ButlerResponse(
            response=self.response + other.response,
            intents=self.intents + [intent + len(self.response) for intent in other.intents],
        )

    @classmethod
    def make(cls, response: str, intent: ButlerIntentEnum | None = None) -> "ButlerResponse":
        intents = [DeepIntent(span=(0, len(response)), intent=intent)] if intent else []
        return ButlerResponse(response=response, intents=intents)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ButlerResponse):
            return False

        return (
            self.response == other.response
            and len(self.intents) == len(other.intents)
            and all(i1 == i2 for i1, i2 in zip(self.intents, other.intents, strict=False))
        )


class Butler(StatefulAgent[[ButlerRequest], ButlerResponse]):
    def __init__(self, categorizer: PromptCategorizer) -> None:
        super().__init__()
        self.categorizer = categorizer

    @register_state(entrypoint=True)
    async def greet(self, input: ButlerRequest) -> ButlerResponse:
        assert input.interaction_type == InteractionType.INVITATION
        self.move_to(self.introduce_conversations)

        return ButlerResponse(response="What's on your mind?")

    @register_state()
    async def introduce_conversations(self, input: ButlerRequest) -> ButlerResponse:
        assert isinstance(input, IntroduceResponsesRequest)

        num_system_models = sum(1 for m in input.models if m.source == "system")
        payload = await self.categorizer.acategorize(input.user_prompt)
        topic = payload.category.lower()

        match num_system_models:
            case 0:
                response = "Here are the AIs you requested. Let me know which you like better."
            case 1:
                response = (
                    f"On the left is the AI you requested. On the right is another I chose to help with {topic}. "
                    "Let me know which you like better."
                )
            case _ if num_system_models >= 2:
                response = (
                    f"It looks like you're asking about {topic}. I've chosen two AIs to help you with that. "
                    "Let me know which you like better. Swipe to the right to see more AIs."
                )

        self.move_to(self.check_errors)

        return ButlerResponse(response=response)

    @register_state()
    async def check_errors(self, input: ButlerRequest) -> ButlerResponse:
        if not isinstance(input, ErrorRequest):
            self.move_to(self.respond_to_user_action)
            return await self.respond_to_user_action(input)  # type: ignore

        self.move_to(self.introduce_conversations)

        return (
            ButlerResponse.make("Yikes! One of the AIs is not working well. Do you want me to ")
            + ButlerResponse.make("get another one", intent=ButlerIntentEnum.RESPONSE_NEW)
            + ButlerResponse.make("?")
        )

    @register_state()
    async def respond_to_user_action(self, input: ButlerRequest) -> ButlerResponse:
        assert isinstance(input, UserActionRequest)
        preferred_model = input.preferred_model.name if input.preferred_model else None

        match input.action:
            case ScratchcardAction.NO_SCRATCHCARD:
                return (
                    ButlerResponse.make(f"You chose {preferred_model}. Continue with it or ")
                    + ButlerResponse.make("start a new topic", intent=ButlerIntentEnum.CHAT_NEW)
                    + ButlerResponse.make(".")
                )
            case ScratchcardAction.SCRATCHCARD:
                return ButlerResponse.make("Great! Here are some credits to keep you going.")
            case ScratchcardAction.AFTER_SCRATCHING:
                return (
                    ButlerResponse.make(f"Continue with {preferred_model} or ")
                    + ButlerResponse.make("start a new topic", intent=ButlerIntentEnum.CHAT_NEW)
                    + ButlerResponse.make(".")
                )

        self.move_to(self.introduce_conversations)
