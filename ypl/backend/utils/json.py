import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        # UUID objects
        if isinstance(obj, UUID):
            return str(obj)

        # Date and time objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, time):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return str(obj)

        # Decimal numbers
        if isinstance(obj, Decimal):
            return str(obj)

        # Enum values
        if isinstance(obj, Enum):
            return obj.value

        # Path objects
        if isinstance(obj, Path):
            return str(obj)

        # Sets
        if isinstance(obj, set):
            return list(obj)

        # Bytes
        if isinstance(obj, bytes):
            return obj.decode("utf-8")

        # Any objects with a to_json method
        if hasattr(obj, "to_json"):
            return obj.to_json()

        # Any objects with a __dict__ attribute
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        return super().default(obj)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, cls=CustomJSONEncoder)
