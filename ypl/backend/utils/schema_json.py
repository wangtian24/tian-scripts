from typing import Any

from pgvector.sqlalchemy import Vector
from pydantic import GetJsonSchemaHandler
from sqlalchemy.dialects.postgresql import TSVECTOR


def vector_json_schema(cls: Any, core_schema: dict, handler: GetJsonSchemaHandler) -> dict[str, str]:
    # Support vector types in openapi schema.
    return {"type": "string", "title": "TSVector"}


# Override the default json schema for vector types, to avoid pydantic failures.
TSVECTOR.__get_pydantic_json_schema__ = classmethod(vector_json_schema)  # type: ignore
Vector.__get_pydantic_json_schema__ = classmethod(vector_json_schema)
