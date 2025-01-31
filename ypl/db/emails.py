import uuid
from typing import TYPE_CHECKING

from sqlmodel import Field

from ypl.db.base import BaseModel

if TYPE_CHECKING:
    pass


# This table stores logs of emails that were successfully sent.
class EmailLogs(BaseModel, table=True):
    __tablename__ = "email_logs"

    email_log_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # Email address of the user who received the email
    email_sent_to: str = Field(nullable=False, index=True)
    # Name of the campaign that was sent
    campaign_name: str = Field(nullable=False)
