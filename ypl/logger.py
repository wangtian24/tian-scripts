import logging
import re

from ypl.backend.config import settings


class RedactingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            redacted_msg = redact_sensitive_data(msg)
            print(redacted_msg)
        except Exception:
            self.handleError(record)


def redact_sensitive_data(text: str) -> str:
    # Email pattern
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    # Redact email addresses
    redacted = re.sub(email_pattern, "[REDACTED EMAIL]", text)

    # TODO: Add more sensitive data patterns to redact

    return redacted


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add our custom redacting handler
    handler = RedactingHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


logger = setup_logger()

if settings.USE_GOOGLE_CLOUD_LOGGING:
    import google.cloud.logging

    client = google.cloud.logging.Client()
    client.setup_logging()
