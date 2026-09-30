"""Chat API request and response schemas."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ChatRequest(BaseModel):
    """Schema for incoming chat query requests."""

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Non-empty chat session identifier",
    )
    message: Optional[str] = Field(
        None,
        max_length=10000,
        description="User query or message content",
    )
    query: Optional[str] = Field(
        None,
        max_length=10000,
        description="Alias for message content",
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("session_id cannot be empty or whitespace-only")
        return v.strip()

    @model_validator(mode="after")
    def validate_message_or_query(self) -> "ChatRequest":
        msg = self.message or self.query
        if not msg or not isinstance(msg, str) or not msg.strip():
            raise ValueError("message cannot be empty or whitespace-only")
        self.message = msg.strip()
        return self


class ChatResponse(BaseModel):
    """Schema for chat response returned by API."""

    session_id: str = Field(
        ...,
        description="Session identifier associated with the response",
    )
    response: str = Field(
        ...,
        description="Assistant response synthesized by the agent",
    )
