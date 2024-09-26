import pytest
from pytest_alembic.config import Config
from pytest_mock_resources import PostgresConfig, create_postgres_fixture


@pytest.fixture
def alembic_config() -> Config:
    """Override this fixture to configure the exact alembic context setup required."""
    return Config()


@pytest.fixture(scope="session")
def pmr_postgres_config() -> PostgresConfig:
    # As we add more extensions, we will need to have a custom image that has all the extensions installed
    return PostgresConfig(image="pgvector/pgvector:pg15")  # type: ignore


alembic_engine = create_postgres_fixture()  # type: ignore
