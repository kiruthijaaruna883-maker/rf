"""Authentication component rendering Login and Registration forms."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import (
    APIClientError,
    AuthenticationError,
    BackendUnavailableError,
    ConflictError,
    RegulatoryChatClient,
    ValidationError,
)
from frontend.config import APP_SUBTITLE, APP_TITLE, REGULATORY_DISCLAIMER
from frontend.session import set_authenticated_user


def render_auth_view(client: RegulatoryChatClient) -> None:
    """Render authentication portal supporting Login and Registration."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(f"# ⚖️ {APP_TITLE}")
        st.markdown(f"*{APP_SUBTITLE}*")
        st.info(REGULATORY_DISCLAIMER, icon="ℹ️")

        tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with tab_login:
            st.markdown("### Sign In to Your Account")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", key="login_username").strip()
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

                if submitted:
                    if not username or not password:
                        st.error("Please enter both username and password.")
                    else:
                        try:
                            token_resp = client.login(username=username, password=password)
                            token = token_resp.get("access_token")
                            if not token:
                                st.error("Login failed: no access token returned.")
                            else:
                                user_info = client.get_me(token=token)
                                set_authenticated_user(token=token, user_info=user_info)
                                st.success("Authentication successful! Loading workspace...")
                                st.rerun()
                        except AuthenticationError:
                            st.error("Incorrect username or password. Please try again.")
                        except BackendUnavailableError as exc:
                            st.error(f"Backend service unavailable: {exc.message}")
                        except APIClientError as exc:
                            st.error(f"Sign in failed: {exc.message}")
                        except Exception as exc:
                            st.error("An unexpected error occurred during sign in.")

        with tab_register:
            st.markdown("### Create New Professional Account")
            with st.form("register_form", clear_on_submit=False):
                reg_user = st.text_input("Username", key="reg_username").strip()
                reg_email = st.text_input("Work Email", key="reg_email").strip()
                reg_pass = st.text_input("Password", type="password", key="reg_password")
                reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
                reg_submitted = st.form_submit_button("Register Account", use_container_width=True)

                if reg_submitted:
                    if not reg_user or not reg_email or not reg_pass:
                        st.error("All registration fields are required.")
                    elif reg_pass != reg_confirm:
                        st.error("Passwords do not match. Please re-enter your password.")
                    elif "@" not in reg_email or "." not in reg_email:
                        st.error("Please provide a valid email address.")
                    else:
                        try:
                            user_resp = client.register(username=reg_user, email=reg_email, password=reg_pass)
                            st.success(
                                f"Account created for '{user_resp.get('username', reg_user)}'! "
                                "Please switch to the 'Sign In' tab to log in."
                            )
                        except ConflictError as exc:
                            st.error(f"Registration conflict: {exc.message}")
                        except ValidationError as exc:
                            st.error(f"Validation error: {exc.message}")
                        except BackendUnavailableError as exc:
                            st.error(f"Backend service unavailable: {exc.message}")
                        except APIClientError as exc:
                            st.error(f"Registration failed: {exc.message}")
                        except Exception as exc:
                            st.error("An unexpected error occurred during registration.")
