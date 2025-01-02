import logging
import os
import re

from google.cloud import logging as google_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers.transports.background_thread import BackgroundThreadTransport

from ypl.backend.config import settings

MAX_LOGGED_FIELD_LENGTH_CHARS = 1000


def redact_sensitive_data(text: str) -> str:
    if settings.ENVIRONMENT == "local":
        return text

    # Email pattern
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    # Redact email addresses
    redacted = re.sub(email_pattern, "[REDACTED EMAIL]", text)

    # TODO: Add more sensitive data patterns to redact

    return redacted


class RedactingMixin:
    def redact_record(self, record: logging.LogRecord) -> None:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_data(record.msg)
        elif isinstance(record.msg, dict):
            record.msg = {k: redact_sensitive_data(str(v)) for k, v in record.msg.items()}

        # Handle extra parameters
        if hasattr(record, "extra"):
            for k, v in record.extra.items():
                setattr(record, k, redact_sensitive_data(str(v)))


class RedactingHandler(RedactingMixin, CloudLoggingHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.redact_record(record)
        super().emit(record)


class RedactingStreamHandler(RedactingMixin, logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.redact_record(record)
        super().emit(record)


class TruncatingMixin:
    def truncate_value(self, value: str) -> str:
        if len(value) > MAX_LOGGED_FIELD_LENGTH_CHARS:
            return value[:MAX_LOGGED_FIELD_LENGTH_CHARS] + "... (truncated)"
        return value

    def truncate_record(self, record: logging.LogRecord) -> None:
        if isinstance(record.msg, str):
            record.msg = self.truncate_value(record.msg)
        elif isinstance(record.msg, dict):
            record.msg = {k: self.truncate_value(str(v)) for k, v in record.msg.items()}

        # Handle extra parameters
        if hasattr(record, "extra"):
            for k, v in record.extra.items():
                setattr(record, k, self.truncate_value(str(v)))


class TruncatingHandler(TruncatingMixin, CloudLoggingHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.truncate_record(record)
        super().emit(record)


class TruncatingStreamHandler(TruncatingMixin, logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.truncate_record(record)
        super().emit(record)


class ConsolidatedHandler(RedactingMixin, TruncatingMixin, CloudLoggingHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.redact_record(record)
        self.truncate_record(record)
        super().emit(record)


class ConsolidatedStreamHandler(RedactingMixin, TruncatingMixin, logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.redact_record(record)
        self.truncate_record(record)
        super().emit(record)


def setup_google_cloud_logging() -> None:
    try:
        client = google_logging.Client()
        name = os.environ.get("GCP_PROJECT_ID") or "default"

        transport_options = {
            "grace_period": 5.0,
            "batch_size": 20,
            "max_latency": 1.0,
            "max_retries": 5,
        }

        consolidated_handler = ConsolidatedHandler(
            client, name=name, transport=BackgroundThreadTransport, transport_options=transport_options
        )

        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(consolidated_handler)

        # Add a fallback handler for when cloud logging fails
        fallback_handler = ConsolidatedStreamHandler()
        fallback_handler.setLevel(logging.WARNING)  # Only log warnings and above to fallback
        root_logger.addHandler(fallback_handler)

        root_logger.propagate = False

    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Google Cloud Logging setup failed: {e}")


if settings.USE_GOOGLE_CLOUD_LOGGING:
    setup_google_cloud_logging()
else:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(ConsolidatedStreamHandler())
    root_logger.propagate = False
