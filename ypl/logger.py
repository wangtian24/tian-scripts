import logging
import os
import re
from datetime import datetime
from typing import Any

import orjson
from dotenv import load_dotenv
from google.cloud import logging as google_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers.transports.background_thread import BackgroundThreadTransport

MAX_LOGGED_FIELD_LENGTH_CHARS = 32000

load_dotenv()


def redact_sensitive_data(text: str) -> str:
    if os.environ.get("ENVIRONMENT") == "local":
        return text

    # Email pattern
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    # Redact email addresses
    redacted = re.sub(email_pattern, "[REDACTED EMAIL]", text)

    # TODO: Add more sensitive data patterns to redact

    return redacted


class RedactingMixin:
    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return redact_sensitive_data(value)
        elif isinstance(value, dict):
            # recursive redact the value in dict, we don't process the key though.
            return {k: self._redact_value(v) for k, v in value.items()}
        elif isinstance(value, list | tuple):
            # recursive redact the value in list or tuple
            return type(value)(self._redact_value(item) for item in value)
        else:
            return value

    def redact_record(self, record: logging.LogRecord) -> None:
        record.msg = self._redact_value(record.msg)

        # Handle extra parameters
        if hasattr(record, "extra"):
            for k, v in record.extra.items():
                setattr(record, k, self._redact_value(v))


class RedactingHandler(RedactingMixin, CloudLoggingHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.redact_record(record)
        super().emit(record)


class RedactingStreamHandler(RedactingMixin, logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.redact_record(record)
        super().emit(record)


class TruncatingMixin:
    def truncate_value(self, value: Any) -> Any:
        if isinstance(value, str):
            # if the log entry is a json, this limit applies to every leaf level string value.
            if len(value) > MAX_LOGGED_FIELD_LENGTH_CHARS:
                return value[:MAX_LOGGED_FIELD_LENGTH_CHARS] + "... (truncated)"
            return value
        elif isinstance(value, dict):
            return {k: self.truncate_value(v) for k, v in value.items()}
        elif isinstance(value, list | tuple):
            return type(value)(self.truncate_value(item) for item in value)
        else:
            return value

    def truncate_record(self, record: logging.LogRecord) -> None:
        record.msg = self.truncate_value(record.msg)

        # Handle extra parameters
        if hasattr(record, "extra"):
            for k, v in record.extra.items():
                setattr(record, k, self.truncate_value(v))


class TruncatingHandler(TruncatingMixin, CloudLoggingHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.truncate_record(record)
        super().emit(record)


class TruncatingStreamHandler(TruncatingMixin, logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.truncate_record(record)
        super().emit(record)


class ConsolidatedMixin:
    def _is_json_string(self, msg: str) -> dict | None:
        """Parse a string as JSON and return the dict if successful, None otherwise."""
        try:
            parsed = orjson.loads(msg)
            return parsed if isinstance(parsed, dict) else None
        except (orjson.JSONDecodeError, TypeError):
            return None

    def _format_message(self, record: logging.LogRecord) -> str | dict:
        """Format the log message with module information."""
        if isinstance(record.msg, dict):
            return {"module": record.module, **record.msg}

        if isinstance(record.msg, str) and record.msg.strip().startswith("{"):
            parsed_msg = self._is_json_string(record.msg)
            if parsed_msg is not None:
                return {"module": record.module, **parsed_msg}

        return record.msg


class ConsolidatedHandler(RedactingMixin, TruncatingMixin, CloudLoggingHandler, ConsolidatedMixin):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            formatted_msg = self._format_message(record)
            record.msg = (
                {"module": record.module, "message": formatted_msg} if isinstance(formatted_msg, str) else formatted_msg
            )
            self.redact_record(record)
            self.truncate_record(record)
            super().emit(record)
        except Exception as e:
            print(f"Error in ConsolidatedHandler: {e}")


class ConsolidatedStreamHandler(RedactingMixin, TruncatingMixin, logging.StreamHandler, ConsolidatedMixin):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            formatted_msg = self._format_message(record)
            message = orjson.dumps(formatted_msg) if isinstance(formatted_msg, dict) else str(formatted_msg)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            record.msg = f"[{timestamp} {record.levelname}] [{record.module}] {message}"

            self.redact_record(record)
            self.truncate_record(record)
            super().emit(record)
        except Exception as e:
            print(f"Error in ConsolidatedStreamHandler: {e}")


def setup_google_cloud_logging() -> None:
    try:
        client = google_logging.Client()
        name = os.environ.get("GCP_PROJECT_ID") or "default"
        consolidated_handler = ConsolidatedHandler(client, name=name, transport=BackgroundThreadTransport)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(consolidated_handler)
        root_logger.propagate = False
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Google Cloud Logging setup failed: {e}")


if os.environ.get("USE_GOOGLE_CLOUD_LOGGING", "").lower() in ("true", "1", "yes", "on"):
    setup_google_cloud_logging()
else:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(ConsolidatedStreamHandler())
    root_logger.propagate = False
