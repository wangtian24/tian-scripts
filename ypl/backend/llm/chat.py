import random
import re
from collections.abc import Generator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import UUID

import pandas as pd
from cachetools.func import ttl_cache
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel as BaseModelV1
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlalchemy import text
from sqlmodel import Session, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.db import get_async_engine, get_engine
from ypl.backend.llm.constants import ACTIVE_MODELS_BY_PROVIDER, PROVIDER_MODEL_PATTERNS, ChatProvider
from ypl.backend.llm.routing.route_data_type import PreferredModel, RoutingPreference
from ypl.db.chats import Chat, ChatMessage, MessageType
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider
from ypl.utils import async_timed_cache

DEFAULT_HIGH_SIM_THRESHOLD = 0.825
DEFAULT_UNIQUENESS_THRESHOLD = 0.75
OPENAI_FT_ID_PATTERN = re.compile(r"^ft:(?P<model>.+?):(?P<organization>.+?)::(?P<id>.+?)$")

YuppMessage = HumanMessage | AIMessage | SystemMessage  # this is needed for proper Pydantic typecasting
YuppMessageRow = list[YuppMessage]


class ModelInfo(BaseModelV1):
    provider: ChatProvider | str
    model: str
    api_key: str
    temperature: float | None = None
    base_url: str | None = None


def get_base_model(chat_llm_cls: type[Any] | None, model: str) -> str:
    if chat_llm_cls == ChatOpenAI and (match := OPENAI_FT_ID_PATTERN.match(model)):
        model = match.group("model")

    return model


def get_canonical_model_name(model: str, provider: ChatProvider) -> str:
    """
    Returns the canonical model name for the given provider. Different providers may assign different names to the
    same model, e.g., HuggingFace vs. Google. This function returns the name that is used by the provider.
    """
    match model, provider:
        case "gemma-2-9b-it", ChatProvider.TOGETHER:
            return "google/gemma-2-9b-it"
        case "nemotron-4-340b-instruct", ChatProvider.NVIDIA:
            return "nvidia/nemotron-4-340b-instruct"
        case "yi-large", ChatProvider.NVIDIA:
            return "01-ai/yi-large"
        case "deepseek-coder-v2", ChatProvider.DEEPSEEK:
            return "deepseek-coder"  # this is not a bug. FE does not have v2 in it
        case "qwen1.5-72b-chat", ChatProvider.TOGETHER:
            return "Qwen/Qwen1.5-72B-Chat"
        case "phi-3-mini-4k-instruct", ChatProvider.MICROSOFT:
            return "microsoft/phi-3-mini-4k-instruct"
        case _:
            return model


@lru_cache(maxsize=1024)
def standardize_provider_name(provider: str) -> str:
    return re.sub(r"\s+", "", provider).lower()


def simple_deduce_original_provider(model: str) -> str:
    for pattern, provider in PROVIDER_MODEL_PATTERNS.items():
        if pattern.match(model):
            return provider

    return model


@ttl_cache(ttl=600)  # 10-min cache
def get_all_pro_models() -> Sequence[str]:
    query = select(LanguageModel.internal_name).where(
        LanguageModel.is_pro.is_(True),  # type: ignore
        LanguageModel.deleted_at.is_(None),  # type: ignore
        LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
    )

    with Session(get_engine()) as session:
        return session.exec(query).all()


@ttl_cache(ttl=600)  # 10-min cache
def get_all_strong_models() -> Sequence[str]:
    query = select(LanguageModel.internal_name).where(
        LanguageModel.is_strong.is_(True),  # type: ignore
        LanguageModel.deleted_at.is_(None),  # type: ignore
        LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
    )

    with Session(get_engine()) as session:
        return session.exec(query).all()


@ttl_cache(ttl=600)  # 10-min cache
def get_user_message(turn_id: str) -> str:
    """Returns the user message for the given turn ID. If no user message is found, returns an empty string."""
    query = select(ChatMessage.content).where(
        ChatMessage.turn_id == UUID(turn_id), ChatMessage.message_type == MessageType.USER_MESSAGE
    )

    with Session(get_engine()) as session:
        return session.exec(query).first() or ""


def get_chat_history(
    chat_id: str,
    turn_id: str | None = None,
    merge_method: str = "single-thread-random",  # noqa: C901
    last_num_turns: int = 100,
) -> list[dict[str, Any]]:
    """
    Returns the chat history for the given chat ID.

    Args:
        chat_id: The ID of the chat to get the history for.
        turn_id: The ID of the last turn to include in the history. If None, the history will include all turns.
        merge_method: The method to use to merge the chat history if there are multiple assistant messages. Currently
            only "single-thread-random" is supported, e.g., randomly select one of the assistant messages if none are
            explicitly selected by the user.

    Returns:
        A sequence of {"content": str, "role": str} objects, where role is one of "user", "assistant", or "quicktake".
    """
    query_dict = dict(chat_id=chat_id)

    if turn_id is None:
        sql_query = text(
            f"""
            SELECT cm.message_type, cm.content, t.turn_id, me.score FROM turns t
                JOIN chat_messages cm ON cm.turn_id = t.turn_id
                LEFT JOIN message_evals me ON me.message_id = cm.message_id
            WHERE t.chat_id = :chat_id
            ORDER BY cm.created_at DESC
            LIMIT {last_num_turns}
            """
        )
    else:
        sql_query = text(
            f"""
            WITH turn_seq_id AS (
                SELECT sequence_id FROM turns WHERE turn_id = :turn_id
            )
            SELECT cm.message_type, cm.content, t.turn_id, me.score FROM turns t
                JOIN chat_messages cm ON cm.turn_id = t.turn_id
                LEFT JOIN message_evals me ON me.message_id = cm.message_id
            WHERE t.chat_id = :chat_id
                AND t.sequence_id BETWEEN (SELECT sequence_id FROM turn_seq_id) - {last_num_turns}
                AND (SELECT sequence_id FROM turn_seq_id)
            ORDER BY cm.created_at DESC
            """
        )
        query_dict["turn_id"] = turn_id

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, query_dict)
        rows = c.fetchall()

    last_turn_id = None
    last_turn_id_to_include = turn_id
    history = []
    asst_buffer: list[dict[str, Any]] = []

    for row in reversed(rows):
        if last_turn_id is not None and last_turn_id_to_include == last_turn_id:
            break

        msg_type, content, turn_id, score = row

        if turn_id != last_turn_id:
            if asst_buffer:
                random.shuffle(asst_buffer)
                history.append(
                    dict(
                        content=max(asst_buffer, key=lambda x: x["score"])["content"],
                        role="assistant",
                    )
                )

            asst_buffer.clear()

        last_turn_id = turn_id

        match msg_type:
            case "USER_MESSAGE":
                history.append(dict(content=content, role="user"))
            case "ASSISTANT_MESSAGE":
                asst_buffer.append(dict(content=content, score=0 if score is None else score))
            case "QUICK_RESPONSE_MESSAGE":
                history.append(dict(content=content, role="quicktake"))
            case _:
                continue

    if asst_buffer:
        random.shuffle(asst_buffer)
        history.append(
            dict(
                content=max(asst_buffer, key=lambda x: x["score"])["content"],
                role="assistant",
            )
        )

    return history


def get_preferences(chat_id: str) -> tuple[RoutingPreference, list[str]]:
    """Returns the preferences and user-selected models for the given chat ID."""
    sql_query = text(
        """
        SELECT cm.assistant_model_name, t.turn_id, e.eval_type, me.score, cm.assistant_selection_source FROM turns t
            JOIN chat_messages cm ON cm.turn_id = t.turn_id
            LEFT JOIN message_evals me ON me.message_id = cm.message_id
            LEFT JOIN evals e ON e.eval_id = me.eval_id
        WHERE t.chat_id = :chat_id AND cm.message_type = 'ASSISTANT_MESSAGE'
        ORDER BY cm.created_at DESC
        LIMIT 100
        """
    )

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, dict(chat_id=chat_id))
        rows = c.fetchall()

    df_rows = []

    for row in reversed(rows):  # reverse to get the oldest turn first
        model_name, turn_id, eval_type, score, selection_src = row
        df_rows.append(
            dict(
                model_name=model_name,
                turn_id=turn_id,
                eval_type=eval_type,
                score=score,
                selection_src=selection_src,
            )
        )

    if not df_rows:
        return RoutingPreference(turns=[]), []

    df = pd.DataFrame(df_rows)
    preferred_models_list = []

    for _, gdf in df.groupby("turn_id", sort=False):
        gdf = gdf[gdf["selection_src"] == "ROUTER_SELECTED"]
        evaluated_models = gdf[gdf["eval_type"].isin(["SELECTION", "ALL_BAD"])]["model_name"].tolist()

        if not evaluated_models:
            preferred_models_list.append(
                PreferredModel(models=gdf["model_name"].tolist(), preferred=None, has_evaluation=False)
            )
            continue

        all_models = gdf["model_name"].tolist()
        preferred_models = gdf[(gdf["eval_type"] == "SELECTION") & (gdf["score"] == 100.0)]["model_name"].tolist()
        preferred_model = preferred_models[0] if preferred_models else None
        preferred_models_list.append(PreferredModel(models=all_models, preferred=preferred_model))

    user_selected_models = list(dict.fromkeys(df[df["selection_src"] == "USER_SELECTED"]["model_name"].tolist()))

    return RoutingPreference(turns=preferred_models_list), user_selected_models


def get_shown_models(turn_id: str) -> list[str]:
    """Returns the models shown to the user for the given turn ID."""
    query = (
        select(ChatMessage.assistant_model_name)
        .where(ChatMessage.turn_id == UUID(turn_id), ChatMessage.message_type == MessageType.ASSISTANT_MESSAGE)
        .order_by(ChatMessage.created_at)  # type: ignore[arg-type]
    )

    with Session(get_engine()) as session:
        return session.exec(query).all()  # type: ignore[return-value]


def get_chat(chat_id: str) -> Chat:
    """Returns the chat for the given chat ID."""
    query = select(Chat).where(Chat.chat_id == chat_id)

    with Session(get_engine()) as session:
        chat = session.exec(query).first()

        if chat is None:
            raise ValueError(f"Chat not found for chat_id: {chat_id}")

        return chat


@ttl_cache(ttl=600)  # 10-min cache
def deduce_semantic_groups(models: tuple[str, ...]) -> dict[str, str]:
    """
    Deduces the semantic group of the given model. If a model is not found in the database, its key is absent from the
    returned dictionary.
    """
    semantic_group_map = {}
    sql_query = text(
        """
        SELECT language_models.internal_name, language_models.semantic_group FROM language_models
        WHERE language_models.internal_name IN :model_names
        AND language_models.deleted_at IS NULL
        AND language_models.status = 'ACTIVE'
        AND language_models.semantic_group IS NOT NULL
        """
    )

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, dict(model_names=models))
        rows = c.fetchall()

    for row in rows:
        model, semantic_group = row[0], row[1]
        semantic_group_map[model] = semantic_group.lower().strip()

    return semantic_group_map


@ttl_cache(ttl=600)  # 10-min cache
def deduce_original_providers(models: tuple[str, ...]) -> dict[str, str]:
    provider_map = {}
    models_left = set(models)

    for model in models_left.copy():
        for pattern, provider in PROVIDER_MODEL_PATTERNS.items():
            if pattern.match(model):
                provider_map[model] = provider
                models_left.remove(model)

    if not models_left:
        # We are done
        return provider_map

    models_left_ = tuple(models_left)

    sql_query = text(
        """
        SELECT providers.name, language_models.internal_name FROM language_models
            JOIN providers ON language_models.provider_id = providers.provider_id
        WHERE language_models.internal_name IN :model_names
        """
    )

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, dict(model_names=models_left_))
        rows = c.fetchall()

        for row in rows:
            provider, model = row[0], row[1]
            models_left.remove(model)
            provider_map[model] = provider

    for model in models_left:
        # On failure, assume model name is provider name
        provider_map[model] = model

    return provider_map


@ttl_cache(ttl=600)  # 10-minute cache
def deduce_original_provider(model: str) -> str:
    for pattern, provider in PROVIDER_MODEL_PATTERNS.items():
        if pattern.match(model):
            return provider

    sql_query = text(
        """
        SELECT providers.name FROM language_models
            JOIN providers ON language_models.provider_id = providers.provider_id
        WHERE language_models.internal_name = :model
        """
    )

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, dict(model=model))
        res = c.first() or (None,)

    name = res[0] or model

    return standardize_provider_name(name)  # return model if all else fails  # type: ignore


@async_timed_cache(seconds=600)
async def adeduce_original_provider(model: str) -> str:
    """Tries to deduce the original provider of a model."""
    # First try to match the model string to a known provider
    for pattern, provider in PROVIDER_MODEL_PATTERNS.items():
        if pattern.match(model):
            return provider

    # If no match was found, try to return the provider in the database
    query = (
        select(Provider.name)
        .join(LanguageModel)
        .where(
            LanguageModel.internal_name == model,
            LanguageModel.deleted_at.is_(None),  # type: ignore
            LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
            Provider.deleted_at.is_(None),  # type: ignore
            Provider.is_active.is_(True),  # type: ignore
        )
        .limit(1)
    )

    async with AsyncSession(get_async_engine()) as session:
        providers = await session.exec(query)

    if provider_ := providers.first():
        return standardize_provider_name(provider_)

    # If all else fails, return the model name
    return model


def get_chat_model(
    info: ModelInfo,
    chat_model_pool: dict[ChatProvider, list[str]] = ACTIVE_MODELS_BY_PROVIDER,
    **chat_kwargs: Any | None,
) -> BaseChatModel:
    """
    Gets the chat model based on the provider and the model name. For OpenAI, Anthropic, Google, Mistral, and HF, the
    only required fields are the provider name, model name, and the API key. For Together, DeepSeek, Nvidia, Anyscale,
    Qwen, and 01, `info` should additionally contain the `base_url` field.
    """
    provider, model, api_key = info.provider, info.model, info.api_key
    chat_kwargs = chat_kwargs.copy()

    if isinstance(provider, str):
        provider = ChatProvider.from_string(provider)

    chat_llms = {
        ChatProvider.OPENAI: ChatOpenAI,
        ChatProvider.ANTHROPIC: ChatAnthropic,
        ChatProvider.GOOGLE: ChatGoogleGenerativeAI,
        ChatProvider.MISTRAL: ChatMistralAI,
        ChatProvider.TOGETHER: ChatOpenAI,
        ChatProvider.DEEPSEEK: ChatOpenAI,
        ChatProvider.NVIDIA: ChatOpenAI,
        ChatProvider.ANYSCALE: ChatOpenAI,
        ChatProvider.ZERO_ONE: ChatOpenAI,
        ChatProvider.QWEN: ChatOpenAI,
        ChatProvider.MICROSOFT: ChatOpenAI,
    }

    chat_llm_cls = chat_llms.get(provider)
    full_model = get_canonical_model_name(model, provider)
    base_model = get_base_model(chat_llm_cls, model)

    if not chat_llm_cls:
        raise ValueError(f"Unsupported provider: {provider}")

    if base_model not in chat_model_pool.get(provider, []):
        raise ValueError(f"Unsupported model: {base_model} for provider: {provider}")

    if info.base_url and "base_url" not in chat_kwargs:
        chat_kwargs["base_url"] = info.base_url

    if full_model.startswith("o1") and provider == ChatProvider.OPENAI:
        chat_kwargs["temperature"] = 1  # temperature must be 1 for o1 models

    return chat_llm_cls(api_key=SecretStr(api_key), model=full_model, **chat_kwargs)  # type: ignore


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


# langchain uses Pydantic v1 in YuppMessage; using for compatibility
class Persona(BaseModelV1):
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


# langchain uses Pydantic v1 in YuppMessage; using for compatibility
class YuppChatMessageHistory(BaseModelV1):
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


ChatUserType = TypeVar("ChatUserType", bound=MultiChatUser)


class YuppChatUserGenerator(Generic[ChatUserType]):
    """Generates chat users."""

    async def agenerate_users(self) -> Generator[ChatUserType, None, None]:
        """Generates chat users asynchronously. Defaults to synchronous implementation if not overriden."""
        return self.generate_users()

    def generate_users(self) -> Generator[ChatUserType, None, None]:
        """Generates chat users."""
        raise NotImplementedError


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


ChatMessageType1 = TypeVar("ChatMessageType1", bound=BaseMessage)
ChatMessageType2 = TypeVar("ChatMessageType2", bound=BaseMessage)


def chat_message_cast_to(message: ChatMessageType1, target_type: type[ChatMessageType2]) -> ChatMessageType2:
    message.type = target_type.schema()["properties"]["type"]["default"]
    return target_type(**message.dict())


def get_db_message_type(message: ChatMessageType1) -> MessageType:
    match message:
        case HumanMessage():
            return MessageType.USER_MESSAGE
        case AIMessage():
            return MessageType.ASSISTANT_MESSAGE
        case _:
            raise ValueError(f"Unsupported message type: {type(message)}")


async def get_turn_id_from_message_id(message_id: UUID) -> UUID | None:
    """Get turn_id from message_id by querying the chat_messages table."""
    async with AsyncSession(get_async_engine()) as session:
        message = await session.get(ChatMessage, message_id)
        if message:
            return message.turn_id
        return None
