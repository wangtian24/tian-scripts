import logging
import os
import re

from google.cloud import logging as google_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers.transports.sync import SyncTransport

from ypl.backend.config import settings


def redact_sensitive_data(text: str) -> str:
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


def setup_google_cloud_logging() -> None:
    try:
        client = google_logging.Client()
        handler = RedactingHandler(client, name=os.environ.get("GCP_PROJECT_ID") or "default", transport=SyncTransport)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Google Cloud Logging setup failed: {e}")


if settings.USE_GOOGLE_CLOUD_LOGGING:
    setup_google_cloud_logging()
else:
    logging.basicConfig(level=logging.INFO)
    handler = RedactingStreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(handler)
