import logging
import uuid

from sqlalchemy import Connection, delete, select
from sqlmodel import Session

from ypl.backend.utils.json import json_dumps
from ypl.db.soul_rbac import RolePermission, SoulPermission, SoulRole, UserRole
from ypl.db.users import User


def create_role(session: Session, name: str, description: str, permissions: list[SoulPermission]) -> uuid.UUID:
    """Helper function to create a role with permissions."""
    role = SoulRole(
        name=name,
        description=description,
    )
    session.add(role)
    session.flush()  # Flush to get the role_id

    for permission in permissions:
        role_permission = RolePermission(role_id=role.role_id, permission=permission)
        session.add(role_permission)

    return role.role_id


async def populate_soul_rbac(connection: Connection) -> None:
    """Populate RBAC tables with initial roles and permissions."""
    with Session(connection) as session:
        try:
            # Create roles with their permissions
            admin_permissions = [
                SoulPermission.READ_USERS,
                SoulPermission.WRITE_USERS,
                SoulPermission.DELETE_USERS,
                SoulPermission.MANAGE_PAYMENT_INSTRUMENTS,
                SoulPermission.VIEW_PAYMENT_INSTRUMENTS,
                SoulPermission.MANAGE_CACHES,
                SoulPermission.VIEW_CACHES,
                SoulPermission.MANAGE_MODEL_PERFORMANCE,
                SoulPermission.VIEW_MODEL_PERFORMANCE,
                SoulPermission.MANAGE_CASHOUT,
                SoulPermission.VIEW_CASHOUT,
            ]

            readonly_permissions = [
                SoulPermission.READ_USERS,
                SoulPermission.VIEW_PAYMENT_INSTRUMENTS,
                SoulPermission.VIEW_CACHES,
                SoulPermission.VIEW_MODEL_PERFORMANCE,
                SoulPermission.VIEW_CASHOUT,
            ]

            # Create roles
            admin_role_id = create_role(
                session, "admin", "Full system administrator with all permissions", admin_permissions
            )

            readonly_role_id = create_role(
                session, "readonly", "Read-only access to view system information", readonly_permissions
            )

            # Get all users
            stmt = select(User).where(User.deleted_at.is_(None))  # type: ignore[attr-defined]
            users = session.exec(stmt).scalars().all()  # type: ignore[call-overload]

            admin_emails = {"pankaj@yupp.ai", "ansuman@yupp.ai", "gcmouli@yupp.ai", "gilad@yupp.ai"}

            for user in users:
                if not user.email:
                    continue

                try:
                    if user.email in admin_emails:
                        # Assign admin role
                        user_role = UserRole(user_id=user.user_id, role_id=admin_role_id)
                        session.add(user_role)
                        log_dict = {
                            "message": "Assigned admin role to user",
                            "user_id": user.user_id,
                            "email": user.email,
                        }
                        logging.info(json_dumps(log_dict))
                    elif user.email.endswith("@yupp.ai"):
                        # Assign readonly role
                        user_role = UserRole(user_id=user.user_id, role_id=readonly_role_id)
                        session.add(user_role)
                        log_dict = {
                            "message": "Assigned readonly role to user",
                            "user_id": user.user_id,
                            "email": user.email,
                        }
                        logging.info(json_dumps(log_dict))

                except Exception as e:
                    log_dict = {
                        "message": "Failed to assign role to user",
                        "user_id": user.user_id,
                        "email": user.email,
                        "error": str(e),
                    }
                    logging.error(json_dumps(log_dict))
                    continue

            session.commit()
            logging.info("Successfully populated RBAC tables")

        except Exception as e:
            log_dict = {
                "message": "Failed to populate RBAC tables",
                "error": str(e),
            }
            logging.error(json_dumps(log_dict))
            session.rollback()
            raise


async def remove_soul_rbac(connection: Connection) -> None:
    """Remove RBAC tables."""
    with Session(connection) as session:
        session.execute(delete(UserRole))
        session.execute(delete(RolePermission))
        session.execute(delete(SoulRole))
        session.commit()
