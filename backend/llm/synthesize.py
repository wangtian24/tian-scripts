import random
import uuid
from datetime import datetime

from backend.llm.constants import FIRST_NAMES, LAST_NAMES
from db.all_models import users


def generate_random_user() -> users.User:
    return users.User(
        id=str(uuid.uuid4()),
        name=f"YF {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        email=f"{uuid.uuid4()}@example.com",
        email_verified=datetime.now(),
    )
