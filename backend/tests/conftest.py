import pytest
from dotenv import load_dotenv

from backend.config import settings


@pytest.fixture(scope="session", autouse=True)
def check_environment() -> None:
    """
    Sanity check to ensure that tests are only run in a local test environment.
    """
    if settings.ENVIRONMENT not in ("test", "local"):
        raise ValueError("Tests should only be run in a local test environment.")


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()
