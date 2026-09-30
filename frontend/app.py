"""Streamlit entry point for Regulatory Affairs Assistant frontend."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import RegulatoryChatClient
from frontend.components.auth_view import render_auth_view
from frontend.components.chat_view import render_chat_view
from frontend.components.sidebar import render_sidebar
from frontend.config import APP_TITLE
from frontend.session import init_session_state, is_authenticated


def main() -> None:
    """Main application loop."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    client = RegulatoryChatClient()

    if not is_authenticated():
        render_auth_view(client)
    else:
        render_sidebar(client)
        render_chat_view(client)


if __name__ == "__main__":
    main()
