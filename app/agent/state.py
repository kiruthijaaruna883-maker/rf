"""LangGraph state schema for Regulatory Affairs Assistant agent."""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from app.memory.redis_memory import ConversationMessage


class AgentState(TypedDict):
    """State schema for the LangGraph agent graph.

    Maintains session identity, user query, conversation history loaded
    from Redis memory, tool selection metadata, and the synthesized response.
    """

    session_id: str
    user_id: Optional[str]
    messages: list[ConversationMessage]
    query: str
    tool_name: Optional[str]
    tool_input: Optional[str]
    tool_output: Optional[Any]
    response: str
