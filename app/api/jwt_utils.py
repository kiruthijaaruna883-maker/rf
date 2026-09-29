"""JWT access-token creation and validation utilities for Regulatory Affairs Assistant."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
import jwt

from app.config import get_settings


def create_access_token(subject: str) -> str:
    """Create an access token for the given subject.

    Args:
        subject: The subject identifier (e.g., username or user ID) to encode in the token.

    Returns:
        The encoded JWT token string.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": subject,
        "exp": expire,
    }
    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token.

    Args:
        token: The encoded JWT token string to decode and validate.

    Returns:
        The decoded payload dictionary.

    Raises:
        jwt.PyJWTError: If the token is invalid, expired, malformed, or incorrectly signed.
    """
    settings = get_settings()
    return jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
        options={"require": ["exp", "sub"]},
    )
