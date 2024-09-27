import re
from collections.abc import Generator
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel as BaseModelV1
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from nltk.tokenize import sent_tokenize, word_tokenize
from pydantic.v1 import SecretStr
from sentence_transformers import SentenceTransformer

from ypl.backend import prompts
from ypl.backend.llm.constants import FRONTEND_MODELS_BY_PROVIDER, ChatProvider
from ypl.backend.llm.utils import combine_short_sentences
from ypl.db.chats import MessageType

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


def get_chat_model(
    info: ModelInfo,
    chat_model_pool: dict[ChatProvider, list[str]] = FRONTEND_MODELS_BY_PROVIDER,
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

    return chat_llm_cls(api_key=SecretStr(api_key), model=full_model, **chat_kwargs)  # type: ignore


def get_chat_history_model(
    info: ModelInfo,
    chat_model_pool: dict[ChatProvider, list[str]] = FRONTEND_MODELS_BY_PROVIDER,
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


def compare_llm_responses(
    provider: ChatProvider | str, model: str, api_key: str, prompt: str, responses: dict[str, str]
) -> BaseMessage:
    llm = get_chat_model(ModelInfo(provider=provider, model=model, api_key=api_key))
    chain = prompts.COMPARE_RESPONSES_PROMPT | llm
    return chain.invoke(input={"prompt": prompt, "responses": responses})


def highlight_llm_similarities(
    provider: ChatProvider | str, model: str, api_key: str, responses: dict[str, str]
) -> BaseMessage:
    llm = get_chat_model(ModelInfo(provider=provider, model=model, api_key=api_key))
    chain = prompts.HIGHLIGHT_SIMILARITIES_PROMPT | llm
    return chain.invoke(input={"prompt": "None", "responses": responses})


def highlight_llm_similarities_with_embeddings(
    response_a: str,
    response_b: str,
    high_sim_threshold: float = DEFAULT_HIGH_SIM_THRESHOLD,
    uniqueness_threshold: float = DEFAULT_UNIQUENESS_THRESHOLD,
) -> dict[str, list[str] | list[dict[str, Any]]]:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences_a = combine_short_sentences(sent_tokenize(response_a))
    sentences_b = combine_short_sentences(sent_tokenize(response_b))

    embeddings_a = model.encode(sentences_a, convert_to_tensor=True)
    embeddings_b = model.encode(sentences_b, convert_to_tensor=True)

    similarities = model.similarity(embeddings_a, embeddings_b)  # type: ignore

    high_similarity_pairs = []
    unique_sentences_a = []
    unique_sentences_b = []

    # Find high-similarity pairs
    for i, row in enumerate(similarities):
        for j, sim in enumerate(row):
            if sim >= high_sim_threshold:
                high_similarity_pairs.append(
                    {
                        "sentence_a": sentences_a[i],
                        "sentence_b": sentences_b[j],
                        "similarity": round(sim.item(), 4),
                    }
                )

    # Find unique sentences
    max_similarities_a = torch.max(similarities, dim=1)
    max_similarities_b = torch.max(similarities, dim=0)

    for i, max_sim in enumerate(max_similarities_a.values):
        if max_sim < uniqueness_threshold:
            unique_sentences_a.append(sentences_a[i])

    for j, max_sim in enumerate(max_similarities_b.values):
        if max_sim < uniqueness_threshold:
            unique_sentences_b.append(sentences_b[j])

    return {
        "high_similarity_pairs": high_similarity_pairs,
        "unique_sentences_a": unique_sentences_a,
        "unique_sentences_b": unique_sentences_b,
    }


def prompt_difficulty(
    prompt: str,
    responses: list[str],
    embedding_similarity_weight: float = 0.5,
    structure_similarity_weight: float = 0.3,
    content_similarity_weight: float = 0.2,
) -> dict[str, Any]:
    # Embedding similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(responses, convert_to_tensor=True)
    similarity_matrix = model.similarity(embeddings, embeddings)  # type: ignore
    n = len(responses)
    avg_embedding_similarity = torch.mean(similarity_matrix[torch.triu_indices(n, n, offset=1)])

    # Structure similarity
    sentence_counts = torch.tensor([len(sent_tokenize(response)) for response in responses])
    avg_sentence_count = torch.mean(sentence_counts.float())
    sentence_count_variance = torch.var(sentence_counts.float())
    structure_similarity = 1 - (sentence_count_variance / avg_sentence_count)
    coeff_variation = torch.sqrt(sentence_count_variance) / avg_sentence_count
    structure_similarity = 1 / (1 + coeff_variation)

    # Content similarity
    word_sets = [set(word_tokenize(response.lower())) for response in responses]
    common_words = set.intersection(*word_sets)
    total_words = set.union(*word_sets)
    content_similarity = len(common_words) / len(total_words)

    prompt_difficulty = (
        (1 - avg_embedding_similarity) * embedding_similarity_weight
        + (1 - structure_similarity) * structure_similarity_weight
        + (1 - content_similarity) * content_similarity_weight
    )

    return {
        "prompt_difficulty": prompt_difficulty.item(),
        "embedding_similarity": avg_embedding_similarity.item(),
        "structure_similarity": structure_similarity.item(),
        "content_similarity": content_similarity,
    }


def prompt_difficulty_by_llm(provider: ChatProvider | str, model: str, api_key: str, prompt: str) -> BaseMessage:
    llm = get_chat_model(ModelInfo(provider=provider, model=model, api_key=api_key))
    chain = prompts.PROMPT_DIFFICULTY_PROMPT | llm
    return chain.invoke(input={"prompt": prompt})


def prompt_difficulty_by_llm_with_responses(
    provider: ChatProvider | str, model: str, api_key: str, prompt: str, responses: dict[str, str]
) -> BaseMessage:
    llm = get_chat_model(ModelInfo(provider=provider, model=model, api_key=api_key))
    chain = prompts.PROMPT_DIFFICULTY_WITH_RESPONSES_PROMPT | llm
    return chain.invoke(input={"prompt": prompt, "responses": responses})


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
