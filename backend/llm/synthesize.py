import asyncio
import hashlib
import logging
import random
import uuid
from collections.abc import Generator
from datetime import datetime
from typing import Any

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel as BaseModelV1
from pydantic.v1.error_wrappers import ValidationError
from sqlmodel import Session
from tqdm.asyncio import tqdm_asyncio

from backend.db import get_engine
from backend.llm.chat import (
    LLMChatAssistant,
    ModelInfo,
    MultiChatUser,
    Persona,
    YuppChatIO,
    YuppChatMessageHistory,
    YuppChatUserGenerator,
    chat_message_cast_to,
    get_chat_history_model,
    get_chat_model,
    get_db_message_type,
)
from backend.llm.constants import ALL_MODELS_BY_PROVIDER, FIRST_NAMES, LAST_NAMES
from backend.prompts import SYNTHESIZER_FIRST_ASSISTANT_PROMPT, SYNTHESIZER_GENERATE_PERSONA_PROMPT
from db.all_models import users
from db.chats import Chat, ChatMessage, EvalType, MessageType, Turn
from db.users import SyntheticBackfillAttributes


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
    num_min_initial_characters: int = 15
    num_min_initial_words: int = 2
    timeout: int = 30  # seconds


class SyntheticYuppChatUser(MultiChatUser):
    def __init__(
        self,
        llm: BaseChatModel,
        persona: Persona | None = None,
        initial_message: str = SYNTHESIZER_FIRST_ASSISTANT_PROMPT,
        num_min_initial_characters: int = 15,
        num_min_initial_words: int = 2,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.persona = persona
        self.initial_message = HumanMessage(initial_message)
        self.num_min_initial_characters = num_min_initial_characters
        self.num_min_initial_words = num_min_initial_words

    def copy(self) -> "SyntheticYuppChatUser":
        """Performs a shallow copy for concurrent generation of chats."""
        return SyntheticYuppChatUser(self.llm, self.persona, str(self.initial_message.content))

    def reset(self) -> None:
        super().reset()
        self.respond(self.initial_message, self.initial_message)

    @property
    def is_initial(self) -> bool:
        assert self.chat_history is not None
        return len(self.chat_history.messages) == 0

    async def areset(self) -> None:
        await super().areset()
        await self.arespond(self.initial_message, self.initial_message)

    def _format_llm_messages(self, *messages: BaseMessage) -> BaseMessage:
        return random.choice(messages)

    def _satisfies_min_length_response(self, response: str) -> bool:
        if not self.is_initial:
            return True

        return len(response.split()) >= self.num_min_initial_words and len(response) >= self.num_min_initial_characters

    def _respond(self, *messages: BaseMessage) -> BaseMessage:
        """Responds to messages from one or more LLMs."""
        assert self.chat_history is not None

        message = self._format_llm_messages(*messages)
        response = self.llm.invoke(
            dict(input=message.content, chat_history=self.chat_history.messages)  # type: ignore
        )

        while not self._satisfies_min_length_response(str(response.content)):
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

        while not self._satisfies_min_length_response(str(response.content)):
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
            timeout=self.config.timeout,
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
                timeout=self.config.timeout,
                chat_model_pool=ALL_MODELS_BY_PROVIDER,
                temperature=0.8,
            )

        return persona_llm, user_llm

    def _iterate_personas_and_llms(self) -> Generator[tuple[Persona | None, BaseChatModel], None, None]:
        persona_llm, user_llm = self._instantiate_models()
        generate_num_personas = self.config.generate_num_personas
        personas = self.config.personas

        if self.config.use_personas:
            if generate_num_personas and persona_llm is not None:
                template = ChatPromptTemplate.from_messages([("user", SYNTHESIZER_GENERATE_PERSONA_PROMPT)])
                run_llm = template | persona_llm | StrOutputParser()
                seed = hashlib.md5(str(random.random()).encode()).hexdigest()[: random.randint(8, 16)]
                lines = [run_llm.invoke(dict(seed=seed)) for _ in range(generate_num_personas)]
                personas = []

                for line in lines:
                    try:
                        personas.append(Persona.parse_raw(line))
                    except ValidationError:
                        logging.warning(f"Failed to parse persona: {line}")
                        continue

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
            yield SyntheticYuppChatUser(
                user_llm_,
                persona=persona,
                num_min_initial_words=self.config.num_min_initial_words,
                num_min_initial_characters=self.config.num_min_initial_characters,
            )


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
        get_chat_history_model(eval_llm1_info, chat_model_pool=ALL_MODELS_BY_PROVIDER, timeout=config.timeout)
    )

    eval_llm2 = LLMChatAssistant(
        get_chat_history_model(eval_llm2_info, chat_model_pool=ALL_MODELS_BY_PROVIDER, timeout=config.timeout)
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


def generate_random_user(**kwargs: Any | None) -> users.User:
    return users.User(
        id=str(uuid.uuid4()),
        name=f"YF {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        email=f"{uuid.uuid4()}@example.com",
        email_verified=datetime.now(),
        **kwargs,
    )


class SQLChatIO(YuppChatIO):
    def __init__(self) -> None:
        self.personas_to_db_users: dict[Persona | None, users.User] = {}
        self.backfill_attributes_data: dict[str, Any] = {}
        self.user_kwargs: dict[str, Any] = {}
        self.session = Session(get_engine())

    def populate_backfill_attributes(
        self,
        synth_config: SynthesizerConfig,
        num_attempted_chats_per_user: int,
        git_commit_sha: str,
    ) -> None:
        num_users = synth_config.generate_num_personas or len(synth_config.personas)

        self.backfill_attributes_data = {
            "num_users": num_users,
            "num_attempted_chats_per_user": num_attempted_chats_per_user,
            "user_llm_model": synth_config.user_llm_name,
            "user_llm_temperature": synth_config.user_llm_temperature,
            "judge_models": [],  # TODO; no judges for now
            "judge_model_temperatures": [],  # TODO; no judges for now
            "git_commit_sha": git_commit_sha,
        }

        backfill = SyntheticBackfillAttributes(**self.backfill_attributes_data)
        self.user_kwargs["backfill_job_id"] = backfill.id
        self.session.add(backfill)
        self.session.commit()

    def append_chat(self, chat: YuppChatMessageHistory) -> "SQLChatIO":
        # A bulk insert would be faster, but this will work for now at O(100K) chats
        if chat.user_persona not in self.personas_to_db_users:
            db_user = generate_random_user(**self.user_kwargs)
            db_user.synthetic_attributes = users.SyntheticUserAttributes(
                user_id=db_user.id,
                persona=chat.user_persona.persona if chat.user_persona else "",
                interests=chat.user_persona.interests if chat.user_persona else [],
                style=chat.user_persona.style if chat.user_persona else "",
            )
            self.personas_to_db_users[chat.user_persona] = db_user
            self.session.add(db_user)

        if not chat.messages:
            self.session.commit()
            return self

        db_user = self.personas_to_db_users[chat.user_persona]
        chat_id = str(uuid.uuid4())
        db_chat = Chat(
            chat_id=chat_id,
            title=chat.messages[0][0].content[:100],
            path=f"/chat/{chat_id}",
            creator_user_id=db_user.id,
            is_public=True,
        )
        self.session.add(db_chat)
        db_turn: Turn | None = None
        turn_no = 0

        for messages in chat.messages:
            if db_turn is None:
                db_turn = Turn(
                    chat_id=db_chat.chat_id,
                    sequence_id=turn_no,
                    creator_user_id=db_user.id,
                )

                turn_no += 1
                self.session.add(db_turn)

            assert db_turn is not None
            eval_data: dict[str, Any] = {}

            for message_idx, message in enumerate(messages):
                chat_message_data: dict[str, Any] = dict(
                    turn_id=db_turn.turn_id,
                    message_type=get_db_message_type(message),
                    content=message.content,
                )

                if chat_message_data["message_type"] == MessageType.ASSISTANT_MESSAGE:
                    chat_message_data["assistant_model_name"] = chat.eval_llms[message_idx]

                db_chat_message = ChatMessage(**chat_message_data)
                self.session.add(db_chat_message)

                if chat_message_data["message_type"] == MessageType.ASSISTANT_MESSAGE:
                    eval_data["user_id"] = db_user.id
                    eval_data["turn_id"] = db_turn.turn_id
                    eval_data["eval_type"] = EvalType.SLIDER_V0
                    eval_data["score_1"] = 50.0  # default score for now
                    eval_data["score_2"] = 50.0  # default score for now

                    if message_idx == 0:  # LLM1
                        eval_data["message_1_id"] = db_chat_message.message_id
                    elif message_idx == 1:  # LLM2
                        eval_data["message_2_id"] = db_chat_message.message_id
                    else:
                        raise ValueError("More than two assistant messages not supported for now")

            if eval_data:
                db_turn = None  # new turn

            self.session.commit()

        return self
