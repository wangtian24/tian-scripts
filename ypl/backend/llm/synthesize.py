import asyncio
import hashlib
import logging
import random
import uuid
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, ValidationError
from sqlmodel import Session
from tqdm.asyncio import tqdm_asyncio

from ypl.backend.db import get_engine
from ypl.backend.llm.constants import (
    ACTIVE_MODELS_BY_PROVIDER,
    ALL_MODELS_BY_PROVIDER,
    FIRST_NAMES,
    LAST_NAMES,
    ChatProvider,
)
from ypl.backend.llm.db_helpers import get_chat_model
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.prompts import SYNTHESIZER_FIRST_ASSISTANT_PROMPT, SYNTHESIZER_GENERATE_PERSONA_PROMPT
from ypl.db.all_models import users
from ypl.db.chats import Chat, ChatMessage, Eval, EvalType, MessageType, Turn
from ypl.db.users import SyntheticBackfillAttributes

ChatMessageType1 = TypeVar("ChatMessageType1", bound=BaseMessage)
ChatMessageType2 = TypeVar("ChatMessageType2", bound=BaseMessage)


class SampleLLMEntry(BaseModel):
    """Represents an LLM to be sampled during synthetic conversation generation"""

    info: ModelInfo
    sample_weight: float = 1  # this LLM will be drawn with probability `sample_weight / sum(all)`


class Persona(BaseModel):
    persona: str = ""
    interests: list[str] = []
    style: str = ""

    def __hash__(self) -> int:
        return hash((self.persona, tuple(self.interests), self.style))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Persona):
            return False

        return (
            self.persona == other.persona
            and tuple(self.interests) == tuple(other.interests)
            and self.style == other.style
        )


# langchain uses Pydantic v1 in BaseMessage; using for compatibility
class SynthesizerConfig(BaseModel):
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


YuppMessage = HumanMessage | AIMessage | SystemMessage  # this is needed for proper Pydantic typecasting


class MultiChatUser:
    """
    Represents a conversational agent capable of responding to one or more messages simultaneously. Keeps track of the
    chat history and various attributes. Each context is associated with a unique chat history.
    """

    def __init__(self) -> None:
        self.chat_history: ChatMessageHistory | None = None

    def copy(self) -> "MultiChatUser":
        """Creates a copy of the chat user."""
        raise NotImplementedError

    @property
    def last_message(self) -> YuppMessage:
        """Returns the last generated message from the synthetic user."""
        assert self.chat_history is not None, "Must be called within the context"
        return self.chat_history.messages[-1]  # type: ignore

    def reset(self) -> None:
        self.chat_history = ChatMessageHistory()

    async def areset(self) -> None:
        self.chat_history = ChatMessageHistory()

    async def __aenter__(self) -> "MultiChatUser":
        await self.areset()
        return self

    def __enter__(self) -> "MultiChatUser":
        self.reset()
        return self

    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        self.reset()

    async def __aexit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        await self.areset()

    def respond(self, *messages: YuppMessage) -> YuppMessage:
        """Responds to messages from one or more LLMs."""
        assert self.chat_history is not None, "Chat history not set. Did you forget to enter the context?"
        return self._respond(*messages)

    def _respond(self, *messages: YuppMessage) -> YuppMessage:
        raise NotImplementedError

    async def arespond(self, *messages: YuppMessage) -> YuppMessage:
        """Responds to a message asynchronously."""
        assert self.chat_history is not None, "Chat history not set. Did you forget to enter the context?"
        return await self._arespond(*messages)

    async def _arespond(self, *messages: YuppMessage) -> YuppMessage:
        raise NotImplementedError


def chat_message_cast_to(message: ChatMessageType1, target_type: type[ChatMessageType2]) -> ChatMessageType2:
    message.type = target_type.schema()["properties"]["type"]["default"]
    return target_type(**message.dict())


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

    def _format_llm_messages(self, *messages: YuppMessage) -> YuppMessage:
        return random.choice(messages)

    def _satisfies_min_length_response(self, response: str) -> bool:
        if not self.is_initial:
            return True

        return len(response.split()) >= self.num_min_initial_words and len(response) >= self.num_min_initial_characters

    def _respond(self, *messages: YuppMessage) -> YuppMessage:
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

    async def _arespond(self, *messages: YuppMessage) -> YuppMessage:
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


ChatUserType = TypeVar("ChatUserType", bound=MultiChatUser)


class YuppChatUserGenerator(Generic[ChatUserType]):
    """Generates chat users."""

    async def agenerate_users(self) -> Generator[ChatUserType, None, None]:
        """Generates chat users asynchronously. Defaults to synchronous implementation if not overriden."""
        return self.generate_users()

    def generate_users(self) -> Generator[ChatUserType, None, None]:
        """Generates chat users."""
        raise NotImplementedError


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


def get_chat_history_model(
    info: ModelInfo,
    chat_model_pool: dict[ChatProvider, list[str]] = ACTIVE_MODELS_BY_PROVIDER,
    **chat_kwargs: Any | None,
) -> BaseChatModel:
    llm = get_chat_model(info, chat_model_pool=chat_model_pool, **chat_kwargs)
    conv_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return conv_template | llm  # type: ignore


YuppMessageRow = list[YuppMessage]


class YuppChatMessageHistory(BaseModel):
    """
    Holds the chat history for a Yupp chat. Each turn can be composed of multiple chat messages (e.g., from two
    LLMs in parallel), so we use a list of messages to represent a turn.
    """

    messages: list[YuppMessageRow] = []
    judgements: list[int | None] = []  # for v1, it is assumed that all judgements are between 1 and 100 inclusive
    eval_llms: list[str] = []
    judge_llm: str | None = None
    user_persona: Persona | None = None
    chat_id: str | None = None

    def initial_prompt_and_responses(self) -> tuple[str | None, Any, list[Any]]:
        """Returns the prompt and respones from the initial turn."""
        return self.chat_id, self.messages[0][0].content, [m.content for m in self.messages[1]]

    def triplet_blocks(self) -> Generator[tuple[YuppMessage, YuppMessage, YuppMessage], None, None]:
        """Generates triplet blocks of user-llm1-llm2 messages, similar to the front-end's behavior."""
        for idx in range(0, (len(self.messages) // 2) * 2, 2):
            if len(self.messages[idx]) != 1 or len(self.messages[idx + 1]) != 2:
                raise ValueError("Each block must have one user message and two LLM messages")

            yield self.messages[idx][0], self.messages[idx + 1][0], self.messages[idx + 1][1]


class LLMChatAssistant(MultiChatUser):
    def __init__(self, llm: BaseChatModel):
        super().__init__()
        self.llm = llm

    def _respond(self, *messages: YuppMessage) -> YuppMessage:
        """Responds to the first message only"""
        assert len(messages) == 1, "Only one message is supported"
        assert self.chat_history is not None

        message = messages[0]
        response = self.llm.invoke(
            dict(input=message.content, chat_history=self.chat_history.messages)  # type: ignore
        )
        self.chat_history.messages.append(message)
        self.chat_history.messages.append(response)

        return response  # type: ignore

    async def _arespond(self, *messages: YuppMessage) -> YuppMessage:
        """Responds to the first message only"""
        assert len(messages) == 1, "Only one message is supported"
        assert self.chat_history is not None

        message = messages[0]
        response = await self.llm.ainvoke(
            dict(input=message.content, chat_history=self.chat_history.messages)  # type: ignore
        )
        self.chat_history.messages.append(message)
        self.chat_history.messages.append(response)

        return response  # type: ignore


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
    llm1_message: YuppMessage | None = None
    llm2_message: YuppMessage | None = None

    try:
        async with sem, user, eval_llm1, eval_llm2:
            response: YuppMessage = user.last_message  # the initial message the user sent
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
    config: SynthesizerConfig,
    user: SyntheticYuppChatUser,
    num_chats: int = 1,
    num_parallel: int = 16,
    chunk_size: int = 1000,
) -> list[YuppChatMessageHistory]:
    """Synthesizes multiple chats asynchronously using the given configuration."""
    sem = asyncio.Semaphore(num_parallel)
    results = []

    # this is to avoid having too many open FDs
    for idx in range(0, num_chats, chunk_size):
        tasks = [asynthesize_chat(config, user.copy(), sem) for _ in range(min(chunk_size, num_chats - idx))]
        results += list(await tqdm_asyncio.gather(*tasks))

    return results


def generate_random_user(**kwargs: Any | None) -> users.User:
    return users.User(
        user_id=str(uuid.uuid4()),
        name=f"YF-{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        email=f"{uuid.uuid4()}@example.com",
        email_verified=datetime.now(),
        **kwargs,
    )


class YuppChatIO:
    def append_chat(self, chat: YuppChatMessageHistory) -> "YuppChatIO":
        """Appends a chat to the writer."""
        raise NotImplementedError

    def write_all_chats(self, chats: list[YuppChatMessageHistory]) -> "YuppChatIO":
        for chat in chats:
            self.append_chat(chat)

        return self

    def read_chats(self) -> list[YuppChatMessageHistory]:
        """Reads chats from the writer."""
        raise NotImplementedError

    def delete(self) -> None:
        """Deletes the object underlying the writer."""
        raise NotImplementedError

    def flush(self) -> None:
        """Flushes the writer."""
        pass


def get_db_message_type(message: ChatMessageType1) -> MessageType:
    match message:
        case HumanMessage():
            return MessageType.USER_MESSAGE
        case AIMessage():
            return MessageType.ASSISTANT_MESSAGE
        case _:
            raise ValueError(f"Unsupported message type: {type(message)}")


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
        judge_models: list[str] | None = None,
        judge_model_temperatures: list[float] | None = None,
    ) -> None:
        num_users = synth_config.generate_num_personas or len(synth_config.personas)
        judge_models_: list[str] = judge_models or []
        judge_model_temperatures_: list[float] = judge_model_temperatures or []

        self.backfill_attributes_data = {
            "num_users": num_users,
            "num_attempted_chats_per_user": num_attempted_chats_per_user,
            "user_llm_model": synth_config.user_llm_name,
            "user_llm_temperature": synth_config.user_llm_temperature,
            "judge_models": judge_models_,
            "judge_model_temperatures": judge_model_temperatures_,
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
                user_id=db_user.user_id,
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
            creator_user_id=db_user.user_id,
            is_public=True,
        )
        self.session.add(db_chat)
        db_turn: Turn | None = None
        turn_no = 0

        for row_idx, messages in enumerate(chat.messages):
            if db_turn is None:
                db_turn = Turn(
                    chat_id=db_chat.chat_id,
                    sequence_id=turn_no,
                    creator_user_id=db_user.user_id,
                )

                turn_no += 1
                self.session.add(db_turn)

            assert db_turn is not None
            eval_data: dict[str, Any] = {}

            # Add judgement data if available.
            has_judgement = chat.judge_llm is not None and (
                len(chat.judgements) > row_idx
                and chat.judgements[row_idx] is not None
                and messages
                and get_db_message_type(messages[0]) == MessageType.ASSISTANT_MESSAGE
            )

            if has_judgement:
                eval_data["judge_model_name"] = chat.judge_llm
                eval_data["score_1"] = float(chat.judgements[row_idx])  # type: ignore # mypy thinks this is unsafe
                eval_data["score_2"] = 100 - chat.judgements[row_idx]  # type:ignore # mypy thinks this is unsafe

            # Populate the messages
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
                    eval_data["user_id"] = db_user.user_id
                    eval_data["turn_id"] = db_turn.turn_id
                    eval_data["eval_type"] = EvalType.SELECTION

                    if message_idx <= 1:  # two LLMs
                        eval_data[f"message_{message_idx + 1}_id"] = db_chat_message.message_id
                    else:
                        raise ValueError("More than two assistant messages not supported for now")

            if eval_data:
                db_turn = None  # new turn

                if has_judgement:
                    self.session.add(Eval(**eval_data))

            self.session.commit()

        return self


class JsonChatIO(YuppChatIO):
    def __init__(self, filename: str) -> None:
        self.path = Path(filename)
        self.chats: list[YuppChatMessageHistory] = []

    def append_chat(self, chat: YuppChatMessageHistory) -> "JsonChatIO":
        self.chats.append(chat)
        return self

    def read_chats(self) -> list[YuppChatMessageHistory]:
        chats = []

        with self.path.open() as f:
            for line in f:
                chats.append(YuppChatMessageHistory.parse_raw(line))

        self.chats = chats
        return self.chats

    def delete(self) -> None:
        self.path.unlink()

    def flush(self, mode: str = "a") -> None:
        with self.path.open(mode=mode) as f:
            for chat in self.chats:
                f.write(chat.json())
                f.write("\n")

        self.chats = []
