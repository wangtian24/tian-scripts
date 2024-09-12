import pytest
import sqlalchemy
from pytest_alembic.config import Config

from backend.config import Settings


@pytest.fixture
def alembic_config() -> Config:
    """Override this fixture to configure the exact alembic context setup required."""
    return Config()


@pytest.fixture
def alembic_engine() -> sqlalchemy.Engine:
    """Override this fixture to provide pytest-alembic powered tests with a database handle."""
    settings = Settings()
    return sqlalchemy.create_engine(settings.db_url)
