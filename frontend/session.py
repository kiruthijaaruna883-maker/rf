"""Streamlit session state management for Regulatory Affairs Assistant."""

from __future__ import annotations

import uuid
from typing import Any, Optional
import streamlit as st


def generate_session_id() -> str:
    """Generate a unique hex session identifier for chat conversations."""
    return uuid.uuid4().hex


def init_session_state() -> None:
    """Initialize required Streamlit session state keys if not already present."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "access_token" not in st.session_state:
        st.session_state["access_token"] = None
    if "user_info" not in st.session_state:
        st.session_state["user_info"] = None
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = generate_session_id()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "auth_error" not in st.session_state:
        st.session_state["auth_error"] = None
    if "auth_success" not in st.session_state:
        st.session_state["auth_success"] = None


def set_authenticated_user(token: str, user_info: dict[str, Any]) -> None:
    """Store authenticated session state upon successful login."""
    st.session_state["authenticated"] = True
    st.session_state["access_token"] = token
    st.session_state["user_info"] = user_info
    st.session_state["auth_error"] = None
    st.session_state["auth_success"] = None


def logout() -> None:
    """Clear credentials and reset session state upon logout."""
    st.session_state["authenticated"] = False
    st.session_state["access_token"] = None
    st.session_state["user_info"] = None
    st.session_state["messages"] = []
    st.session_state["session_id"] = generate_session_id()
    st.session_state["auth_error"] = None
    st.session_state["auth_success"] = None


def start_new_session() -> str:
    """Create a new session ID and clear visible messages for the new conversation."""
    new_id = generate_session_id()
    st.session_state["session_id"] = new_id
    st.session_state["messages"] = []
    return new_id


def clear_current_messages() -> None:
    """Clear messages in the current conversation."""
    st.session_state["messages"] = []


def add_chat_message(role: str, content: str) -> None:
    """Append a validated message to the visible session conversation history."""
    if not isinstance(role, str) or role not in ("user", "assistant"):
        raise ValueError(f"Invalid message role: {role!r}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Message content cannot be empty.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"role": role, "content": content.strip()})


def get_messages() -> list[dict[str, str]]:
    """Return the list of conversation messages for the active session."""
    return list(st.session_state.get("messages", []))


def get_current_session_id() -> str:
    """Return the current conversation session identifier."""
    if "session_id" not in st.session_state or not st.session_state.get("session_id"):
        st.session_state["session_id"] = generate_session_id()
    return st.session_state["session_id"]


def is_authenticated() -> bool:
    """Check whether the current session is authenticated."""
    return bool(st.session_state.get("authenticated", False) and st.session_state.get("access_token"))


def get_access_token() -> Optional[str]:
    """Return the current JWT access token if authenticated."""
    return st.session_state.get("access_token")


def get_current_user() -> Optional[dict[str, Any]]:
    """Return authenticated user profile dictionary if available."""
    return st.session_state.get("user_info")
