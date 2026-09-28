"""Authentication security and password utilities for Regulatory Affairs Assistant."""

import bcrypt


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt with a fresh salt.

    Args:
        password: The plaintext password string to hash.

    Returns:
        The bcrypt hash string (UTF-8 decoded).
    """
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed_bytes.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash.

    Args:
        password: The plaintext password string to verify.
        password_hash: The stored bcrypt hash string.

    Returns:
        True if the password matches the hash, False otherwise.
    """
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False
