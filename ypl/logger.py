from ypl.backend.config import settings

if settings.USE_GOOGLE_CLOUD_LOGGING:
    import google.cloud.logging

    client = google.cloud.logging.Client()
    client.setup_logging()
