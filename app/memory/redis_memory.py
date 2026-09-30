"""
Redis-backed conversation memory for chat sessions.

Stores chronological conversation messages per session with bounded history
and TTL-based expiration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import redis
from redis.exceptions import RedisError

from app.config import get_settings

logger = logging.getLogger(__name__)

ALLOWED_ROLES: set[str] = {"user", "assistant"}
DEFAULT_KEY_PREFIX: str = "chat:memory"
DEFAULT_MAX_MESSAGES: int = 50
DEFAULT_TTL_SECONDS: int = 86400  # 24 hours


class RedisMemoryError(RedisError):
    """Exception raised for Redis conversation memory errors."""
    pass


@dataclass
class ConversationMessage:
    """Internal representation of a conversation message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in ALLOWED_ROLES:
            raise ValueError(
                f"Invalid role: {self.role!r}. Allowed roles are: {', '.join(sorted(ALLOWED_ROLES))}"
            )
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Message content must be a non-empty string.")

    def to_dict(self) -> dict[str, str]:
        """Convert message to dictionary format."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationMessage:
        """Create ConversationMessage from dictionary."""
        if not isinstance(data, dict):
            raise ValueError("Message data must be a dictionary.")
        if "role" not in data or "content" not in data:
            raise ValueError("Message data missing required 'role' or 'content' keys.")
        return cls(role=data["role"], content=data["content"])


class RedisConversationMemory:
    """
    Redis-backed conversation memory store for chat sessions.

    Maintains message lists keyed by session ID with chronological order,
    bounded history, and TTL-based expiration.
    """

    def __init__(
        self,
        client: Optional[redis.Redis] = None,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        max_messages: Optional[int] = DEFAULT_MAX_MESSAGES,
        ttl_seconds: Optional[int] = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.key_prefix = key_prefix.strip(":")
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds

        if client is not None:
            self._client = client
        else:
            settings = get_settings()
            self._client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD or None,
                decode_responses=True,
            )

    @property
    def client(self) -> redis.Redis:
        """Return the underlying Redis client."""
        return self._client

    def _get_key(self, session_id: str) -> str:
        """Validate session_id and construct the Redis key."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Session ID must be a non-empty string.")
        return f"{self.key_prefix}:{session_id.strip()}"

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Append a validated message to the session conversation history in Redis.

        Preserves chronological ordering, trims history to max_messages if configured,
        and refreshes key TTL.
        """
        key = self._get_key(session_id)
        msg = ConversationMessage(role=role, content=content)
        payload = json.dumps(msg.to_dict())

        try:
            self._client.rpush(key, payload)
            if self.max_messages is not None and self.max_messages > 0:
                self._client.ltrim(key, -self.max_messages, -1)
            if self.ttl_seconds is not None and self.ttl_seconds > 0:
                self._client.expire(key, self.ttl_seconds)
        except RedisError as e:
            logger.error("Redis error adding message to session %s: %s", session_id, e)
            raise RedisMemoryError(f"Failed to add message to Redis memory: {e}") from e

    def get_messages(
        self,
        session_id: str,
    ) -> list[ConversationMessage]:
        """
        Retrieve messages for a session in chronological order.

        Returns an empty list if no messages exist.
        """
        key = self._get_key(session_id)

        try:
            raw_messages = self._client.lrange(key, 0, -1)
        except RedisError as e:
            logger.error("Redis error fetching messages for session %s: %s", session_id, e)
            raise RedisMemoryError(f"Failed to retrieve messages from Redis memory: {e}") from e

        if not raw_messages:
            return []

        messages: list[ConversationMessage] = []
        for idx, raw in enumerate(raw_messages):
            try:
                data = json.loads(raw)
                messages.append(ConversationMessage.from_dict(data))
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(
                    "Malformed message in session %s at index %d: %s",
                    session_id,
                    idx,
                    e,
                )
                raise RedisMemoryError(
                    f"Malformed message in session {session_id} at index {idx}: {e}"
                ) from e

        return messages

    def clear(
        self,
        session_id: str,
    ) -> None:
        """
        Clear conversation history for the specified session key only.
        """
        key = self._get_key(session_id)

        try:
            self._client.delete(key)
        except RedisError as e:
            logger.error("Redis error clearing session %s: %s", session_id, e)
            raise RedisMemoryError(f"Failed to clear Redis memory for session {session_id}: {e}") from e
