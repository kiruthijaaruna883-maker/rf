"""Chat interface component for submitting questions and viewing responses and evidence."""

from __future__ import annotations

import re
import streamlit as st

from frontend.api_client import (
    APIClientError,
    AuthenticationError,
    BackendUnavailableError,
    RateLimitError,
    RegulatoryChatClient,
)
from frontend.config import APP_TITLE
from frontend.session import (
    add_chat_message,
    get_access_token,
    get_current_session_id,
    get_messages,
    logout,
)


def _extract_sources(content: str) -> list[str]:
    """Extract source citation blocks from assistant response text."""
    sources = []
    # Pattern matching [Source N: ...] blocks
    matches = re.findall(r"(\[Source\s+\d+:[^\]]+\])", content, re.IGNORECASE)
    if matches:
        sources.extend(matches)

    # Pattern matching openFDA record headers
    if "openfda" in content.lower() and "drug labeling" in content.lower():
        sources.append("Official openFDA Regulatory Drug Labeling Database (api.fda.gov)")

    return sources


def render_chat_view(client: RegulatoryChatClient) -> None:
    """Render main chat stream, history, and input controls."""
    st.markdown(f"## {APP_TITLE}")
    st.caption("Submit queries regarding regulatory guidance, stability conditions, drug labeling, or user access.")

    # Render conversation history
    messages = get_messages()
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                sources = _extract_sources(content)
                if sources:
                    with st.expander("📚 Sources & Regulatory References", expanded=False):
                        for s in sources:
                            st.markdown(f"- `{s}`")

    # Chat input box
    user_query = st.chat_input(
        "Ask a regulatory question (e.g., 'What are the stability testing conditions for Zone IV?')..."
    )

    if user_query:
        clean_query = user_query.strip()
        if clean_query:
            # 1. Display user query immediately
            add_chat_message("user", clean_query)
            with st.chat_message("user"):
                st.markdown(clean_query)

            # 2. Invoke backend assistant
            token = get_access_token()
            session_id = get_current_session_id()

            if not token:
                st.error("Authentication expired. Please sign in again.")
                logout()
                st.rerun()
                return

            with st.chat_message("assistant"):
                with st.spinner("Consulting regulatory knowledge base and synthesizing response..."):
                    try:
                        resp_data = client.send_chat(
                            token=token,
                            session_id=session_id,
                            message=clean_query,
                        )
                        assistant_text = resp_data.get("response", "")
                        st.markdown(assistant_text)

                        # Check for sources
                        sources = _extract_sources(assistant_text)
                        if sources:
                            with st.expander("📚 Sources & Regulatory References", expanded=False):
                                for s in sources:
                                    st.markdown(f"- `{s}`")

                        add_chat_message("assistant", assistant_text)
                    except AuthenticationError:
                        st.error("Your session has expired. Please sign in again.")
                        logout()
                        st.rerun()
                    except RateLimitError as exc:
                        st.warning(f"Rate limit reached: {exc.message}")
                    except BackendUnavailableError as exc:
                        st.error(f"Backend unavailable: {exc.message}")
                    except APIClientError as exc:
                        st.error(f"Query processing error: {exc.message}")
                    except Exception as exc:
                        st.error("An unexpected error occurred while communicating with the assistant.")
            st.rerun()
