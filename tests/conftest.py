import pytest
from pytest_alembic.config import Config
from pytest_mock_resources import create_postgres_fixture


@pytest.fixture
def alembic_config() -> Config:
    """Override this fixture to configure the exact alembic context setup required."""
    return Config()


alembic_engine = create_postgres_fixture()  # type: ignore
