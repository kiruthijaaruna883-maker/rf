"""Sidebar component displaying user metadata, session controls, and disclaimers."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClientError, RegulatoryChatClient
from frontend.config import REGULATORY_DISCLAIMER
from frontend.session import (
    clear_current_messages,
    get_access_token,
    get_current_session_id,
    get_current_user,
    logout,
    start_new_session,
)


def render_sidebar(client: RegulatoryChatClient) -> None:
    """Render the application sidebar with session management controls."""
    with st.sidebar:
        st.markdown("## ⚖️ Regulatory Assistant")

        # User profile display
        user = get_current_user()
        if user:
            username = user.get("username", "Unknown")
            email = user.get("email", "")
            st.markdown(f"**User:** `{username}`")
            if email:
                st.caption(f"Email: {email}")
        st.divider()

        # Session information
        session_id = get_current_session_id()
        st.markdown("**Active Session:**")
        st.code(session_id, language="text")
        st.divider()

        # Session actions
        st.markdown("### Conversation Actions")

        if st.button("➕ New Conversation", use_container_width=True):
            new_id = start_new_session()
            st.toast(f"Started new session: {new_id[:8]}...", icon="✨")
            st.rerun()

        if st.button("🗑️ Clear Current Chat", use_container_width=True):
            token = get_access_token()
            if token and session_id:
                try:
                    client.clear_chat(token=token, session_id=session_id)
                except APIClientError as exc:
                    st.warning(f"Note: Memory clear notice: {exc.message}")
            clear_current_messages()
            st.toast("Current conversation cleared.", icon="🧹")
            st.rerun()

        st.divider()

        if st.button("🚪 Sign Out", use_container_width=True):
            logout()
            st.toast("Signed out successfully.", icon="👋")
            st.rerun()

        st.divider()
        with st.expander("ℹ️ Regulatory Disclaimer", expanded=False):
            st.caption(REGULATORY_DISCLAIMER)
