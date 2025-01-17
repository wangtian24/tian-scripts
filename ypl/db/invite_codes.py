import uuid
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel

if TYPE_CHECKING:
    from ypl.db.users import User


class SpecialInviteCodeState(Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    FULLY_CLAIMED = "FULLY_CLAIMED"


class SpecialInviteCode(BaseModel, table=True):
    __tablename__ = "special_invite_codes"

    special_invite_code_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)

    code: str = Field(nullable=False, index=True, unique=True)

    # Creator relationship
    creator_user_id: str = Field(foreign_key="users.user_id", nullable=False, index=True)
    creator: "User" = Relationship(
        back_populates="created_special_invite_codes",
    )

    state: SpecialInviteCodeState = Field(nullable=False, default=SpecialInviteCodeState.ACTIVE)

    # How many times this codes can be used in total. If null, this code is unlimited.
    # Note that this field does not represent the usage left
    usage_limit: int | None = Field(nullable=True)

    claim_logs: list["SpecialInviteCodeClaimLog"] = Relationship(back_populates="special_invite_code")


class SpecialInviteCodeClaimLog(BaseModel, table=True):
    __tablename__ = "special_invite_code_claim_logs"
    __table_args__ = (UniqueConstraint("special_invite_code_id", "user_id"),)

    special_invite_code_claim_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)

    special_invite_code_id: uuid.UUID = Field(foreign_key="special_invite_codes.special_invite_code_id", nullable=False)
    special_invite_code: "SpecialInviteCode" = Relationship(back_populates="claim_logs")

    user_id: str = Field(foreign_key="users.user_id", nullable=False)
    user: "User" = Relationship(back_populates="special_invite_code_claim_log")

    ip_address: str = Field(nullable=True)
    user_agent: str = Field(nullable=True)

    # Other additional information that may be useful for abuse detection.
    client_device_info: dict[str, str] = Field(default_factory=dict, sa_type=JSONB, nullable=True)
