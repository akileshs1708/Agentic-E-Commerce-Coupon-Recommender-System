"""
Shared database connection utilities.
All services use this module to obtain SQLAlchemy sessions.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://coupon_user:coupon_pass@postgres:5432/coupon_db"
)

# Create engine with connection pool settings suitable for microservices
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # test connections before using them
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist. Called at service startup."""
    from shared.models import Base
    Base.metadata.create_all(bind=engine)
