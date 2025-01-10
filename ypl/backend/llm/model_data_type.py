from pydantic import BaseModel, SecretStr

from ypl.backend.llm.constants import ChatProvider


class ModelInfo(BaseModel):
    provider: ChatProvider | str
    model: str
    api_key: str | SecretStr
    temperature: float | None = None
    base_url: str | None = None
