import os

import nltk
import pytest
from dotenv import load_dotenv

from ypl.backend.config import settings


@pytest.fixture(scope="session", autouse=True)
def check_environment() -> None:
    """
    Sanity check to ensure that tests are only run in a local test environment.
    """
    if settings.ENVIRONMENT not in ("test", "local"):
        raise ValueError("Tests should only be run in a local test environment.")


@pytest.fixture(scope="session", autouse=True)
def load_env() -> None:
    load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def setup_nltk() -> None:
    nltk_data_path = os.getenv("NLTK_DATA")
    if nltk_data_path:
        nltk.data.path.append(nltk_data_path)
