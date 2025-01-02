from pydantic import BaseModel

from ypl.backend.llm.constants import ChatProvider


class ModelInfo(BaseModel):
    provider: ChatProvider | str
    model: str
    api_key: str
    temperature: float | None = None
    base_url: str | None = None
