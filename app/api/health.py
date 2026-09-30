"""Operational health and readiness probe endpoint for Regulatory Affairs Assistant."""

from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, Response, status
import redis
from sqlalchemy import text

from app.config import get_settings
from app.database.session import engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


def check_database() -> str:
    """Check PostgreSQL database connectivity."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "healthy"
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return "unhealthy"


def check_redis() -> str:
    """Check Redis cache and session memory connectivity."""
    try:
        settings = get_settings()
        client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD or None,
            decode_responses=True,
            socket_timeout=2.0,
        )
        if client.ping():
            return "healthy"
        return "unhealthy"
    except Exception as exc:
        logger.error("Redis health check failed: %s", exc)
        return "unhealthy"


@router.get(
    "/health",
    summary="Operational health and readiness check",
    description=(
        "Public health probe verifying operational status of the API, "
        "PostgreSQL database, and Redis memory store."
    ),
)
def health_check(response: Response) -> dict[str, Any]:
    """Return component and overall operational health status."""
    db_status = check_database()
    redis_status = check_redis()

    is_healthy = (db_status == "healthy" and redis_status == "healthy")

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "database": db_status,
        "redis": redis_status,
    }
