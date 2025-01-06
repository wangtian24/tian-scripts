import json
import os
import secrets
import warnings
from functools import cached_property
from typing import Annotated, Any, Literal, Self

import sqlalchemy
from pydantic import (
    AnyUrl,
    BeforeValidator,
    PostgresDsn,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from ypl.backend.utils.json import json_dumps

DEFAULT_UNSAFE_PASSWORD = "changethis"

CorsOrigins = list[AnyUrl] | str | Literal["*"]


def parse_cors(v: Any) -> list[str]:
    if isinstance(v, str):
        urls = [i.strip() for i in v.split(",") if i]
        return urls
    elif isinstance(v, list):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")
    API_PREFIX: str = "/api"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    X_API_KEY: str = ""

    AWS_REGION_NAME: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ALIBABA_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    ANYSCALE_API_KEY: str = ""
    AZURE_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    FIREWORKS_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    PERPLEXITY_API_KEY: str = ""
    TOGETHERAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    CEREBRAS_API_KEY: str = ""

    DOMAIN: str = "localhost"
    ENVIRONMENT: Literal["test", "local", "staging", "production"] = "local"
    PROJECT_NAME: str = ""
    BACKEND_CORS_ORIGINS: Annotated[CorsOrigins, BeforeValidator(parse_cors)] = []

    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "test")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "test")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost:5432")
    POSTGRES_HOST_NON_POOLING: str = os.getenv("POSTGRES_HOST_NON_POOLING", "localhost:5432")
    POSTGRES_DATABASE: str = os.getenv("POSTGRES_DATABASE", "postgres")
    # For direct DB connection through Cloud SQL Proxy
    # Ref: https://cloud.google.com/sql/docs/postgres/connect-run#connect
    # Looks like "/cloudsql/<INSTANCE_CONNECTION_NAME>"
    CLOUD_SQL_PROXY_INSTANCE_UNIX_SOCKET: str = ""

    CACHE_DIR: str = ".cache"
    USE_GOOGLE_CLOUD_LOGGING: bool = True

    ROUTING_GOOD_MODELS_RANK_THRESHOLD: int = 3  # the rank cutoff for what is considered a "good" model for routing
    ROUTING_GOOD_MODELS_ALWAYS: bool = False  # if true, a good model will always be included in the selected models
    ROUTING_DO_LOGGING: bool = True  # if true, logging will be done

    ROUTING_ERROR_FILTER_SOFT_THRESHOLD: float | None = None
    ROUTING_ERROR_FILTER_HARD_THRESHOLD: float | None = None
    ROUTING_ERROR_FILTER_SOFT_REJECT_PROB: float | None = None

    # If provided, the ranked router will use a metarouting strategy among the provided policies as keys. For a list
    # of the available policies, see `ypl.backend.llm.routing.policy.SelectionCriteria`. If this is empty or null,
    # the router will use the default policy.
    ROUTING_WEIGHTS: dict[str, float] = {}

    # The timeout for the prompt categorizer in routing
    ROUTING_TIMEOUT_SECS: float = 1.25

    # Whether to use prompt-conditional routing. Defaults to false.
    ROUTING_USE_PROMPT_CONDITIONAL: bool = False
    ROUTING_REPUTABLE_PROVIDERS: list[str] = ["openai", "google", "anthropic", "azure", "microsoft", "meta"]
    OPENAI_API_KEY_ROUTING: str = ""

    # The GCP storage path to the prompt categorizer model.
    CATEGORIZER_MODEL_PATH: str = "gs://yupp-models/online-classifier-base.zip"

    # PyTorch Serve service settings
    PYTORCH_SERVE_GCP_SERVICE_NAME: str = "backend-pytorch-service"
    PYTORCH_SERVE_GCP_REGION: str = "us-central1"  # only region that supports L4 GPUs
    PYTORCH_SERVE_GCP_URL: str = ""

    # The cloud run project ID and region
    GCP_PROJECT_ID: str = ""
    GCP_REGION: str = "us-east4"
    GCP_REGION_GEMINI_2: str = "us-central1"

    CDP_API_KEY_NAME: str = os.getenv("CDP_API_KEY_NAME", "")
    CDP_API_KEY_PRIVATE_KEY: str = os.getenv("CDP_API_KEY_PRIVATE_KEY", "")

    BASESCAN_API_KEY: str = os.getenv("BASESCAN_API_KEY", "")
    BASESCAN_API_URL: str = os.getenv("BASESCAN_API_URL", "")

    CRYPTO_WALLET_PATH: str = os.getenv("CRYPTO_WALLET_PATH", ".")
    CRYPTO_EXCHANGE_PRICE_API_URL_COINBASE: str = os.getenv("CRYPTO_EXCHANGE_PRICE_API_URL_COINBASE", "")
    CRYPTO_EXCHANGE_PRICE_API_URL_COINGECKO: str = os.getenv("CRYPTO_EXCHANGE_PRICE_API_URL_COINGECKO", "")

    DEFAULT_QT_TIMEOUT_SECS: float = 1.5

    # The base URL of the yupp-head app, set to staging by default.
    # Example use case: when updating models on yupp-mind, we need to revalidate the model caches on yupp-head too.
    YUPP_HEAD_APP_BASE_URL: str = "https://chaos.yupp.ai"

    def _get_gcp_secret(self, secret_name: str) -> str:
        """Retrieve secret from Google Cloud Secret Manager."""

        from google.cloud import secretmanager

        try:
            if not self.GCP_PROJECT_ID:
                return ""

            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.GCP_PROJECT_ID}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8") or ""
        except Exception as e:
            import logging

            logging.error(
                json_dumps(
                    {
                        "message": "Error retrieving secret from Google Cloud Secret Manager",
                        "error": str(e),
                        "secret_name": secret_name,
                    }
                )
            )

            return ""

    @computed_field  # type: ignore[misc]
    @cached_property
    def axis_upi_config(self) -> dict:
        return json.loads(self._get_gcp_secret(f"axis-upi-config-{self.ENVIRONMENT}"))  # type: ignore[no-any-return]

    @computed_field  # type: ignore[misc]
    @property
    def server_host(self) -> str:
        # Use HTTPS for anything other than local development
        if self.ENVIRONMENT == "local":
            return f"http://{self.DOMAIN}"
        return f"https://{self.DOMAIN}"

    def _db_url(self, async_mode: bool) -> str:
        scheme = "postgresql" + ("+asyncpg" if async_mode else "")
        return PostgresDsn.build(
            scheme=scheme,
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_HOST,
            path=f"{self.POSTGRES_DATABASE}",
        ).unicode_string()

    def _db_url_direct(self, async_mode: bool) -> str:
        drivername = "postgresql" + ("+asyncpg" if async_mode else "")
        return sqlalchemy.engine.url.URL.create(
            drivername=drivername,
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            database=self.POSTGRES_DATABASE,
            query={"host": f"{self.CLOUD_SQL_PROXY_INSTANCE_UNIX_SOCKET}" + ("/.s.PGSQL.5432" if async_mode else "")},
        ).render_as_string(hide_password=False)

    @computed_field  # type: ignore[misc]
    @property
    def db_ssl_mode(self) -> str:
        return "disable" if self.ENVIRONMENT == "local" else "require"

    @computed_field  # type: ignore[misc]
    @property
    def db_url(self) -> str:
        return self._db_url(async_mode=False)

    @computed_field  # type: ignore[misc]
    @property
    def db_url_async(self) -> str:
        return self._db_url(async_mode=True)

    @computed_field  # type: ignore[misc]
    @property
    def db_url_direct(self) -> str:
        return self._db_url_direct(async_mode=False)

    @computed_field  # type: ignore[misc]
    @property
    def db_url_direct_async(self) -> str:
        return self._db_url_direct(async_mode=True)

    def _check_default_secret(self, var_name: str, value: str | None) -> None:
        if value == DEFAULT_UNSAFE_PASSWORD:
            message = (
                f'The value of {var_name} is "{DEFAULT_UNSAFE_PASSWORD}", '
                "for security, please change it, at least for deployments."
            )
            if self.ENVIRONMENT == "local":
                warnings.warn(message, stacklevel=1)
            else:
                raise ValueError(message)

    @model_validator(mode="after")
    def _enforce_non_default_secrets(self) -> Self:
        self._check_default_secret("OPENAI_API_KEY", self.SECRET_KEY)
        self._check_default_secret("AWS_ACCESS_KEY_ID", self.AWS_ACCESS_KEY_ID)
        self._check_default_secret("AWS_SECRET_ACCESS_KEY", self.AWS_SECRET_ACCESS_KEY)
        self._check_default_secret("ALIBABA_API_KEY", self.ALIBABA_API_KEY)
        self._check_default_secret("ANTHROPIC_API_KEY", self.ANTHROPIC_API_KEY)
        self._check_default_secret("ANYSCALE_API_KEY", self.ANYSCALE_API_KEY)
        self._check_default_secret("AZURE_API_KEY", self.AZURE_API_KEY)
        self._check_default_secret("DEEPSEEK_API_KEY", self.DEEPSEEK_API_KEY)
        self._check_default_secret("FIREWORKS_API_KEY", self.FIREWORKS_API_KEY)
        self._check_default_secret("GOOGLE_API_KEY", self.GOOGLE_API_KEY)
        self._check_default_secret("HUGGINGFACE_API_KEY", self.HUGGINGFACE_API_KEY)
        self._check_default_secret("MISTRAL_API_KEY", self.MISTRAL_API_KEY)
        self._check_default_secret("NVIDIA_API_KEY", self.NVIDIA_API_KEY)
        self._check_default_secret("OPENROUTER_API_KEY", self.OPENROUTER_API_KEY)
        self._check_default_secret("PERPLEXITY_API_KEY", self.PERPLEXITY_API_KEY)
        self._check_default_secret("TOGETHERAI_API_KEY", self.TOGETHERAI_API_KEY)

        return self

    @model_validator(mode="after")
    def validate_db_config(self) -> Self:
        if self.ENVIRONMENT in ["production", "staging"]:
            # Only validate during actual runtime, not during tests
            if os.getenv("PYTEST_CURRENT_TEST") is None:  # This env var is automatically set by pytest
                test_values = ["test", "postgres", "localhost:5432"]
                if (
                    self.POSTGRES_USER in test_values
                    or self.POSTGRES_PASSWORD == "test"
                    or self.POSTGRES_HOST in test_values
                    or self.POSTGRES_DATABASE in test_values
                ):
                    raise ValueError(
                        f"Database configuration using test values in {self.ENVIRONMENT} environment. "
                        "Please set proper database credentials."
                    )
        return self


settings = Settings()
