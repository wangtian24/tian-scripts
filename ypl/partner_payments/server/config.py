import json
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Reuse the same env file as the backend on local.
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    ENVIRONMENT: Literal["test", "local", "staging", "production"] = "local"
    PROJECT_NAME: str = "Yupp Partner Payments Server"
    API_PREFIX: str = "/api"
    X_API_KEY: str = ""

    GCP_PROJECT_ID: str = ""


class SecretManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _get_gcp_secret(self, secret_name: str) -> str:
        """Retrieve secret from Google Cloud Secret Manager."""

        import json
        import logging

        from google.api_core import exceptions as core_exceptions
        from google.api_core import retry
        from google.cloud import secretmanager

        try:
            log_dict = {
                "message": "Retrieving secret from Google Cloud Secret Manager",
                "secret_name": secret_name,
            }
            logging.info(json.dumps(log_dict))
            if not self.settings.GCP_PROJECT_ID:
                raise ValueError("GCP_PROJECT_ID is not set")

            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.settings.GCP_PROJECT_ID}/secrets/{secret_name}/versions/latest"

            retry_config = retry.Retry(
                initial=0.5,
                maximum=10.0,
                multiplier=2.0,
                predicate=retry.if_exception_type(
                    core_exceptions.ResourceExhausted,
                    core_exceptions.DeadlineExceeded,
                    core_exceptions.ServiceUnavailable,
                ),
                deadline=30.0,
            )

            # Each retry will wait for 2x the previous wait time, up to a maximum of 10 seconds.
            # Each retry has a timeout of 5 seconds, and the entire request has a timeout of 30 seconds.
            response = client.access_secret_version(request={"name": name}, retry=retry_config, timeout=5.0)
            data = response.payload.data.decode("UTF-8")
            logging.info(
                json.dumps(
                    {
                        "message": "Secret retrieved successfully",
                        "secret_name": secret_name,
                    }
                )
            )
            return data
        except Exception as e:
            logging.error(
                json.dumps(
                    {
                        "message": "Error retrieving secret from Google Cloud Secret Manager",
                        "error": str(e),
                        "secret_name": secret_name,
                    }
                )
            )

            return ""

    async def get_axis_upi_config(self) -> dict:
        return json.loads(self._get_gcp_secret(f"axis-upi-config-{self.settings.ENVIRONMENT}"))  # type: ignore[no-any-return]

    async def get_tabapay_config(self) -> dict:
        return json.loads(self._get_gcp_secret(f"tabapay-config-{self.settings.ENVIRONMENT}"))  # type: ignore[no-any-return]


settings = Settings()
secret_manager = SecretManager(settings)
