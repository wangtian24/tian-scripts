import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column, Text, UniqueConstraint
from sqlalchemy import Enum as sa_Enum
from sqlmodel import Field, Relationship, SQLModel

from ypl.db.base import BaseModel

if TYPE_CHECKING:
    from ypl.db.users import User


class SoulPermission(str, enum.Enum):
    # User management permissions
    READ_USERS = "read_users"
    WRITE_USERS = "write_users"
    DELETE_USERS = "delete_users"

    # Payment related permissions
    MANAGE_PAYMENT_INSTRUMENTS = "manage_payment_instruments"
    VIEW_PAYMENT_INSTRUMENTS = "view_payment_instruments"

    # System level permissions
    MANAGE_CACHES = "manage_caches"
    VIEW_CACHES = "view_caches"

    # Model management permissions
    MANAGE_MODEL_PERFORMANCE = "manage_model_performance"
    VIEW_MODEL_PERFORMANCE = "view_model_performance"

    # Cashout related permissions
    MANAGE_CASHOUT = "manage_cashout"
    VIEW_CASHOUT = "view_cashout"


class RolePermission(SQLModel, table=True):
    """Association model for role-permission relationship."""

    __tablename__ = "soul_role_permissions"

    role_id: uuid.UUID = Field(
        foreign_key="soul_roles.role_id",
        primary_key=True,
        nullable=False,
    )
    permission: SoulPermission = Field(
        sa_column=Column(
            sa_Enum(SoulPermission, name="soul_permission_enum"),
            primary_key=True,
            nullable=False,
        )
    )

    # Back reference to role
    role: "SoulRole" = Relationship(back_populates="role_permissions")


# Association table for user-role many-to-many relationship
class UserRole(SQLModel, table=True):
    """Association model for user-role relationship."""

    __tablename__ = "soul_user_roles"

    user_id: str = Field(
        foreign_key="users.user_id",
        primary_key=True,
        nullable=False,
    )
    role_id: uuid.UUID = Field(
        foreign_key="soul_roles.role_id",
        primary_key=True,
        nullable=False,
    )


class SoulRole(BaseModel, table=True):
    """Model for storing user roles and their associated permissions."""

    __tablename__ = "soul_roles"

    role_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    name: str = Field(sa_type=Text, unique=True, nullable=False)
    description: str = Field(sa_type=Text, nullable=False)

    # Relationships
    users: list["User"] = Relationship(
        sa_relationship_kwargs={"secondary": "soul_user_roles"}, back_populates="soul_roles"
    )
    role_permissions: list[RolePermission] = Relationship(back_populates="role")

    class Config:
        arbitrary_types_allowed = True

    __table_args__ = (UniqueConstraint("name", name="uq_soul_roles_name"),)

    def __repr__(self) -> str:
        return f"<Role {self.name}>"

    @property
    def permissions(self) -> list[SoulPermission]:
        """Get list of permissions for this role."""
        return [rp.permission for rp in self.role_permissions]
