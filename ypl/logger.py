from ypl.backend.config import settings

if settings.USE_GOOGLE_CLOUD_LOGGING:
    import google.cloud.logging

    client = google.cloud.logging.Client()  # type: ignore
    client.setup_logging()  # type: ignore
