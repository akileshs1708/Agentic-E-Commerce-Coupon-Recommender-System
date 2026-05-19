"""
Database migration script using Alembic.
Run this if you need to apply schema changes to an existing database.

Usage:
  # From project root inside the api container:
  docker-compose exec api python -m alembic upgrade head

  # Or generate a new migration after model changes:
  docker-compose exec api python -m alembic revision --autogenerate -m "description"
"""
import sys
sys.path.insert(0, "/app")

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from shared.models import Base
from shared.database import DATABASE_URL

config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
