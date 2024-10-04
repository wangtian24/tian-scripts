import sys
from logging.config import fileConfig

import alembic_postgresql_enum  # noqa: F401 for side-effect to handle enum definition in alembic.
import sqlmodel
from alembic import context
from sqlalchemy import engine_from_config, pool

from ypl.backend.config import Settings
from ypl.db.all_models import all_models  # noqa: F401 for populating metadata.

config = context.config
settings = Settings()

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)


target_metadata = sqlmodel.SQLModel.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=settings.db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        user_module_prefix="sqlmodel.sql.sqltypes.",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = str(settings.db_url)

    connectable = context.config.attributes.get("connection", None)

    if connectable is None:
        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            user_module_prefix="sqlmodel.sql.sqltypes.",
        )

        with context.begin_transaction():
            context.run_migrations()


print(f"Current POSTGRES_HOST: {settings.POSTGRES_HOST}")
if ("upgrade" in sys.argv or "downgrade" in sys.argv) and settings.ENVIRONMENT.lower() != "local":
    print(f"WARNING: You are about to upgrade the {settings.ENVIRONMENT.upper()} database!")
    approval = input("Type 'yupp' to continue: ").strip().lower()
    if approval != "yupp":
        print("Migration aborted!")
        exit(1)

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

config.set_main_option("sqlalchemy.url", settings.db_url.replace("%", "%%"))
