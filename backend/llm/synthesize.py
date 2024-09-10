import asyncio
import logging
import random
import uuid
from collections.abc import Generator
from datetime import datetime

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel as BaseModelV1
from tqdm.asyncio import tqdm_asyncio

from backend.llm.chat import (
    LLMChatAssistant,
    ModelInfo,
    MultiChatUser,
    Persona,
    YuppChatMessageHistory,
    YuppChatUserGenerator,
    chat_message_cast_to,
    get_chat_history_model,
    get_chat_model,
)
from backend.llm.constants import ALL_MODELS_BY_PROVIDER, FIRST_NAMES, LAST_NAMES
from backend.prompts import SYNTHESIZER_FIRST_ASSISTANT_PROMPT, SYNTHESIZER_GENERATE_PERSONA_PROMPT
from db.all_models import users


class SampleLLMEntry(BaseModelV1):
    """Represents an LLM to be sampled during synthetic conversation generation"""

    info: ModelInfo
    sample_weight: float = 1  # this LLM will be drawn with probability `sample_weight / sum(all)`


# langchain uses Pydantic v1 in BaseMessage; using for compatibility
class SynthesizerConfig(BaseModelV1):
    personas: list[Persona] = []
    use_personas: bool = True  # if False, use unconditional generation
    generate_num_personas: int = 0  # if zero, do not generate any personas and use the ones specified

    persona_llm_provider: str = "openai"
    persona_llm_name: str = "gpt-4o-mini"
    persona_llm_api_key: str = ""

    user_llm_provider: str = "openai"
    user_llm_name: str = "gpt-4o-mini"
    user_llm_temperature: float = 1.0
    user_llm_api_key: str = ""

    eval_llms: list[SampleLLMEntry] = []

    num_turns_min: int = 2
    num_turns_max: int = 2
    timeout_ms: int = 30000


class SyntheticYuppChatUser(MultiChatUser):
    def __init__(
        self,
        llm: BaseChatModel,
        persona: Persona | None = None,
        initial_message: str = SYNTHESIZER_FIRST_ASSISTANT_PROMPT,
    ):
        super().__init__()
        self.llm = llm
        self.persona = persona
        self.initial_message = HumanMessage(initial_message)

    def copy(self) -> "SyntheticYuppChatUser":
        """Performs a shallow copy for concurrent generation of chats."""
        return SyntheticYuppChatUser(self.llm, self.persona, str(self.initial_message.content))

    def reset(self) -> None:
        super().reset()
        self.respond(self.initial_message, self.initial_message)

    async def areset(self) -> None:
        await super().areset()
        await self.arespond(self.initial_message, self.initial_message)

    def _format_llm_messages(self, *messages: BaseMessage) -> BaseMessage:
        return random.choice(messages)

    def _respond(self, *messages: BaseMessage) -> BaseMessage:
        """Responds to messages from one or more LLMs."""
        assert self.chat_history is not None

        message = self._format_llm_messages(*messages)
        response = self.llm.invoke(
            dict(input=message.content, chat_history=self.chat_history.messages)  # type: ignore
        )
        self.chat_history.messages.append(message)
        self.chat_history.messages.append(chat_message_cast_to(response, HumanMessage))

        return chat_message_cast_to(response, HumanMessage)

    async def _arespond(self, *messages: BaseMessage) -> BaseMessage:
        """Responds to messages from one or more LLMs."""
        assert self.chat_history is not None

        message = self._format_llm_messages(*messages)
        response = await self.llm.ainvoke(
            dict(input=message.content, chat_history=self.chat_history.messages)  # type: ignore
        )
        self.chat_history.messages.append(message)
        self.chat_history.messages.append(chat_message_cast_to(response, HumanMessage))

        return chat_message_cast_to(response, HumanMessage)


class SyntheticUserGenerator(YuppChatUserGenerator[SyntheticYuppChatUser]):
    def __init__(self, config: SynthesizerConfig):
        self.config = config

    def _instantiate_models(self) -> tuple[BaseChatModel | None, BaseChatModel]:
        """Instantiates the LLMs"""

        user_llm = get_chat_model(
            ModelInfo(
                provider=self.config.user_llm_provider,
                model=self.config.user_llm_name,
                api_key=self.config.user_llm_api_key,
            ),
            timeout=self.config.timeout_ms,
            chat_model_pool=ALL_MODELS_BY_PROVIDER,
            temperature=self.config.user_llm_temperature,
        )
        if not self.config.persona_llm_api_key:
            persona_llm = None
        else:
            persona_llm = get_chat_model(
                ModelInfo(
                    provider=self.config.user_llm_provider,
                    model=self.config.user_llm_name,
                    api_key=self.config.user_llm_api_key,
                ),
                timeout=self.config.timeout_ms,
                chat_model_pool=ALL_MODELS_BY_PROVIDER,
            )

        return persona_llm, user_llm

    def _iterate_personas_and_llms(self) -> Generator[tuple[Persona | None, BaseChatModel], None, None]:
        persona_llm, user_llm = self._instantiate_models()
        generate_num_personas = self.config.generate_num_personas
        personas = self.config.personas

        if self.config.use_personas:
            if generate_num_personas and persona_llm is not None:
                if generate_num_personas > 50:
                    logging.warning("Number of personas to generate should be less than 50.")

                template = ChatPromptTemplate.from_messages([("system", SYNTHESIZER_GENERATE_PERSONA_PROMPT)])
                run_llm = template | persona_llm | StrOutputParser()
                lines = run_llm.invoke(dict(num_personas=generate_num_personas)).splitlines()
                personas = [Persona.parse_raw(line) for line in lines]

            for persona in personas:
                persona_json_str = persona.json().replace(":", ": ").replace(",", ", ")
                persona_json_str = persona_json_str.replace("{", "{{").replace("}", "}}")

                user_template = ChatPromptTemplate.from_messages(
                    [
                        ("system", persona_json_str),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )

                yield persona, user_template | user_llm  # type: ignore
        else:
            user_template = ChatPromptTemplate.from_messages(
                [MessagesPlaceholder("chat_history"), ("human", "{input}")]
            )

            yield None, user_template | user_llm  # type: ignore

    def generate_users(self) -> Generator[SyntheticYuppChatUser, None, None]:
        """Generates synthetic users."""
        for persona, user_llm_ in self._iterate_personas_and_llms():
            yield SyntheticYuppChatUser(user_llm_, persona=persona)


async def asynthesize_chat(
    config: SynthesizerConfig,
    user: SyntheticYuppChatUser,
    sem: asyncio.Semaphore | None = None,
) -> YuppChatMessageHistory:
    """Synthesizes a single chat using the given configuration."""
    sem = sem or asyncio.Semaphore(1)
    sample_distn = np.array([x.sample_weight for x in config.eval_llms])
    sample_distn /= sample_distn.sum()

    eval_llm1_info, eval_llm2_info = np.random.choice(
        np.array([x.info for x in config.eval_llms], dtype=object), 2, p=sample_distn, replace=False
    )

    eval_llm1 = LLMChatAssistant(
        get_chat_history_model(eval_llm1_info, chat_model_pool=ALL_MODELS_BY_PROVIDER, timeout=config.timeout_ms)
    )

    eval_llm2 = LLMChatAssistant(
        get_chat_history_model(eval_llm2_info, chat_model_pool=ALL_MODELS_BY_PROVIDER, timeout=config.timeout_ms)
    )

    num_turns = random.randint(config.num_turns_min, config.num_turns_max)

    messages = []
    llm1_message: BaseMessage | None = None
    llm2_message: BaseMessage | None = None

    try:
        async with sem, user, eval_llm1, eval_llm2:
            response: BaseMessage = user.last_message  # the initial message the user sent
            messages.append([response])

            for turn_idx in range(num_turns - 1):  # minus 1 because we already have the initial prompt
                if turn_idx > 0:
                    assert llm1_message is not None and llm2_message is not None
                    response = await user.arespond(llm1_message, llm2_message)
                    messages.append([response])

                llm1_message, llm2_message = await asyncio.gather(
                    eval_llm1.arespond(response), eval_llm2.arespond(response)
                )
                assert llm1_message is not None and llm2_message is not None
                messages.append([llm1_message, llm2_message])
    except:  # noqa
        pass

    return YuppChatMessageHistory(
        messages=messages, eval_llms=[eval_llm1_info.model, eval_llm2_info.model], user_persona=user.persona
    )


async def asynthesize_chats(
    config: SynthesizerConfig, user: SyntheticYuppChatUser, num_chats: int = 1, num_parallel: int = 16
) -> list[YuppChatMessageHistory]:
    """Synthesizes multiple chats asynchronously using the given configuration."""
    sem = asyncio.Semaphore(num_parallel)
    tasks = [asynthesize_chat(config, user.copy(), sem) for _ in range(num_chats)]

    return list(await tqdm_asyncio.gather(*tasks))


def generate_random_user() -> users.User:
    return users.User(
        id=str(uuid.uuid4()),
        name=f"YF {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        email=f"{uuid.uuid4()}@example.com",
        email_verified=datetime.now(),
    )
