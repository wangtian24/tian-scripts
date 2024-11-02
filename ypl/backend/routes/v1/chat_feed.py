import logging

from fastapi import APIRouter, HTTPException, Query

from ypl.backend.llm import chat_feed
from ypl.backend.llm.chat_feed import ChatWithTurns
from ypl.backend.utils.json import json_dumps

router = APIRouter()


@router.get("/chat-feed", response_model=list[chat_feed.ChatWithTurns])
async def get_chat_feed(
    page: int = Query(0, ge=0, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
) -> list[ChatWithTurns]:
    try:
        return await chat_feed.get_chat_feed(page, page_size)

    except Exception as e:
        log_dict = {
            "message": f"Error load chat feed - page : {page} page_size {page_size}",
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
