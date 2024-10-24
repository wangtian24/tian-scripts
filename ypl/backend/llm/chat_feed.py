import logging
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_serializer
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlmodel.ext.asyncio.session import AsyncSession
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl.backend.db import get_async_engine
from ypl.db.chats import Chat, ChatMessage, MessageType, Turn


class UserSchema(BaseModel):
    user_id: str
    name: str
    image: str | None = None


class LanguageModelSchema(BaseModel):
    language_model_id: uuid.UUID
    name: str
    avatar_url: str | None = None


class ChatMessageSchema(BaseModel):
    message_id: uuid.UUID
    message_type: MessageType
    content: str
    assistant_language_model: LanguageModelSchema | None = None

    # to make sure enum name is returned instead of value (default behaviour)
    @field_serializer("message_type")
    def serialize_message_type(self, message_type: MessageType, _info: Any) -> str:
        return message_type.name


class TurnSchema(BaseModel):
    turn_id: uuid.UUID
    chat_messages: list[ChatMessageSchema]


class ChatWithTurns(BaseModel):
    chat_id: uuid.UUID
    title: str | None = None
    path: str | None = None
    created_at: datetime | None = None
    creator: UserSchema
    turns: list[TurnSchema]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def get_chat_feed(
    page: int,
    page_size: int,
) -> list[ChatWithTurns]:
    query = (
        select(Chat)
        .options(
            selectinload(Chat.turns)  # type: ignore
            .selectinload(Turn.chat_messages)  # type: ignore
            .selectinload(ChatMessage.assistant_language_model),  # type: ignore
            selectinload(Chat.creator),  # type: ignore
        )
        .where(Chat.is_public.is_(True), Chat.deleted_at.is_(None))  # type: ignore
        .order_by(Chat.created_at.desc())  # type: ignore
        .offset((page) * page_size)
        .limit(page_size)
    )

    async with AsyncSession(get_async_engine()) as session:
        result = await session.exec(query)  # type: ignore

    chats = result.scalars().unique().all()

    # Convert the ORM objects to Pydantic models
    chat_feed = [
        ChatWithTurns(
            **chat.model_dump(exclude={"turns", "creator"}),
            creator=UserSchema(**chat.creator.model_dump()),
            turns=[
                TurnSchema(
                    **turn.model_dump(exclude={"chat_messages"}),
                    chat_messages=[
                        ChatMessageSchema(
                            **message.model_dump(exclude={"assistant_language_model"}),
                            assistant_language_model=LanguageModelSchema(
                                **message.assistant_language_model.model_dump()
                            )
                            if message.assistant_language_model
                            else None,
                        )
                        for message in turn.chat_messages
                    ],
                )
                for turn in chat.turns
            ],
        )
        for chat in chats
    ]

    return chat_feed
