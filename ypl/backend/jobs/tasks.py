import asyncio
import logging

from fast_langdetect import detect
from sqlmodel import Session, select

from ypl.backend.db import get_engine
from ypl.backend.jobs.app import init_celery
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, LanguageCode

celery_app = init_celery()


@celery_app.task(max_retries=3)
def store_language_code(chat_message_id: str, content: str) -> None:
    if not content or not chat_message_id:
        return

    try:
        # 'detect' requires a single line.
        detected_language = detect(content.replace("\n", " ").strip())
        lang_code = detected_language["lang"]
        with Session(get_engine()) as session:
            message = session.exec(select(ChatMessage).where(ChatMessage.message_id == chat_message_id)).first()
            if message:
                message.language_code = LanguageCode(lang_code)
                session.commit()
                logging.debug(f"Language code stored for message {chat_message_id}")
            else:
                logging.error(f"Message not found for message_id {chat_message_id}")
    except Exception as e:
        log_dict = {
            "message": "Language detection failed",
            "chat_message_id": chat_message_id,
            "content": content,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))


@celery_app.task
def post_to_slack_task(message: str | None = None, webhook_url: str | None = None, blocks: list | None = None) -> None:
    asyncio.get_event_loop().run_until_complete(post_to_slack(message, webhook_url, blocks))
