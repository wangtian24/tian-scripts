import asyncio
import logging
import os

from sqlmodel import Session, select

from ypl.backend.db import get_engine
from ypl.backend.jobs.app import init_celery
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, LanguageCode

celery_app = init_celery()

models_downloaded = False


def maybe_init_langdetect() -> None:
    global models_downloaded
    if not models_downloaded:
        import fast_langdetect
        from fast_langdetect import detect

        import ypl.db.all_models  # noqa

        # fast_langdetect hardcodes its cache directory.
        # Prevent multiple instances from using the same cache by appending the process ID.
        fast_langdetect.ft_detect.infer.FTLANG_CACHE += f"-pid={os.getpid()}"
        # Force the model to be downloaded by calling detect() for the first time.
        detect("hello")
        models_downloaded = True


@celery_app.task(max_retries=3)
def celery_store_language_code(chat_message_id: str, content: str) -> None:
    store_language_code(chat_message_id, content)


def store_language_code(chat_message_id: str, content: str) -> None:
    maybe_init_langdetect()
    from fast_langdetect import detect

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
                logging.info(f"Language code {lang_code} stored for message {chat_message_id}")
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


async def astore_language_code(chat_message_id: str, content: str) -> None:
    await asyncio.to_thread(store_language_code, chat_message_id, content)


@celery_app.task
def post_to_slack_task(message: str | None = None, webhook_url: str | None = None, blocks: list | None = None) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop and loop.is_running():
            asyncio.ensure_future(post_to_slack(message, webhook_url, blocks))
        else:
            asyncio.run(post_to_slack(message, webhook_url, blocks))
    except Exception as e:
        log_dict = {"message": "Error posting to Slack", "error": str(e)}
        logging.error(json_dumps(log_dict))
