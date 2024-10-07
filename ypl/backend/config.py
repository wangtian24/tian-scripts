import secrets
import warnings
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AnyUrl,
    BeforeValidator,
    PostgresDsn,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    AWS_REGION_NAME: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    OPENAI_API_KEY: str = ""

    DOMAIN: str = "localhost"
    ENVIRONMENT: Literal["test", "local", "staging", "production"] = "local"
    PROJECT_NAME: str = ""
    BACKEND_CORS_ORIGINS: Annotated[CorsOrigins, BeforeValidator(parse_cors)] = []

    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    POSTGRES_HOST: str = ""
    POSTGRES_HOST_NON_POOLING: str = ""
    POSTGRES_DATABASE: str = ""

    CACHE_DIR: str = ".cache"
    USE_GOOGLE_CLOUD_LOGGING: bool = False

    ROUTING_GOOD_MODELS_RANK_THRESHOLD: int = 3  # the rank cutoff for what is considered a "good" model for routing
    ROUTING_GOOD_MODELS_ALWAYS: bool = False  # if true, a good model will always be included in the selected models
    ROUTING_DO_LOGGING: bool = True  # if true, logging will be done

    # If provided, the ranked router will use a metarouting strategy among the provided policies as keys. For a list
    # of the available policies, see `ypl.backend.llm.routing.policy.SelectionCriteria`. If this is empty or null,
    # the router will use the default policy.
    ROUTING_WEIGHTS: dict[str, float] = {}

    # Whether to use prompt-conditional routing. Defaults to false.
    ROUTING_USE_PROMPT_CONDITIONAL: bool = False
    OPENAI_API_KEY_ROUTING: str = ""
    CATEGORIZER_MODEL_PATH: str = ""

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
        return self


settings = Settings()
