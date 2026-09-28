"""Database engine and session management for Regulatory Affairs Assistant."""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings


def _get_app_settings() -> Settings:
    try:
        return get_settings()
    except Exception:
        # Fallback allowing database session module to be imported and inspected
        # without requiring JWT_SECRET_KEY to be set in the environment.
        return Settings(JWT_SECRET_KEY="placeholder-for-db-session-init")


settings = _get_app_settings()

database_url: URL = URL.create(
    drivername="postgresql+psycopg",
    username=settings.POSTGRES_USER,
    password=settings.POSTGRES_PASSWORD or None,
    host=settings.POSTGRES_HOST,
    port=settings.POSTGRES_PORT,
    database=settings.POSTGRES_DB,
)

engine = create_engine(database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and reliably close it in finally."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
