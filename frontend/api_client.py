"""API client for interacting with the FastAPI Regulatory Affairs Assistant backend."""

from __future__ import annotations

import logging
from typing import Any, Optional
import httpx

from frontend.config import API_BASE_URL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Base exception for API client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthenticationError(APIClientError):
    """Exception raised on 401 Unauthorized responses."""
    pass


class ForbiddenError(APIClientError):
    """Exception raised on 403 Forbidden responses."""
    pass


class ConflictError(APIClientError):
    """Exception raised on 409 Conflict responses (e.g., existing user/email)."""
    pass


class ValidationError(APIClientError):
    """Exception raised on 422 Unprocessable Content / Validation errors."""
    pass


class RateLimitError(APIClientError):
    """Exception raised on 429 Rate Limit responses."""
    pass


class BackendUnavailableError(APIClientError):
    """Exception raised on connection failure, timeout, 502 Bad Gateway, or 503 Service Unavailable."""
    pass


class ServerError(APIClientError):
    """Exception raised on 500 Internal Server Error."""
    pass


def _extract_detail(data: Any, default_message: str) -> str:
    """Extract clean error detail message from FastAPI error response."""
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
        elif isinstance(detail, list):
            parts = []
            for item in detail:
                if isinstance(item, dict) and "msg" in item:
                    loc = " -> ".join(str(x) for x in item.get("loc", []) if x != "body")
                    msg = item["msg"]
                    parts.append(f"{loc}: {msg}" if loc else msg)
                else:
                    parts.append(str(item))
            if parts:
                return "; ".join(parts)
    return default_message


class RegulatoryChatClient:
    """HTTP client for FastAPI Regulatory Affairs Assistant backend endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = (base_url or API_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._external_client = client

    def _get_client(self) -> httpx.Client:
        if self._external_client is not None:
            return self._external_client
        return httpx.Client(timeout=self.timeout)

    def _handle_response_error(self, response: httpx.Response, action: str) -> None:
        status_code = response.status_code
        try:
            data = response.json()
        except Exception:
            data = None

        detail = _extract_detail(data, f"{action} failed with HTTP status {status_code}.")

        if status_code == 401:
            raise AuthenticationError(detail, status_code=401)
        elif status_code == 403:
            raise ForbiddenError(detail, status_code=403)
        elif status_code == 409:
            raise ConflictError(detail, status_code=409)
        elif status_code == 422:
            raise ValidationError(detail, status_code=422)
        elif status_code == 429:
            raise RateLimitError(detail, status_code=429)
        elif status_code in (502, 503):
            raise BackendUnavailableError(detail, status_code=status_code)
        elif status_code >= 500:
            raise ServerError(detail, status_code=status_code)
        else:
            raise APIClientError(detail, status_code=status_code)

    def login(self, username: str, password: str) -> dict[str, str]:
        """Authenticate user credentials and return JWT access token."""
        clean_user = username.strip() if isinstance(username, str) else ""
        if not clean_user or not password:
            raise ValidationError("Username and password cannot be empty.")

        url = f"{self.base_url}/auth/login"
        form_data = {"username": clean_user, "password": password}

        try:
            if self._external_client is not None:
                response = self._external_client.post(url, data=form_data)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, data=form_data)
        except httpx.TimeoutException as exc:
            raise BackendUnavailableError("Login request timed out. Please try again.") from exc
        except (httpx.ConnectError, httpx.RequestError) as exc:
            raise BackendUnavailableError(
                f"Unable to connect to backend server at {self.base_url}."
            ) from exc

        if response.status_code != 200:
            self._handle_response_error(response, "Login")

        try:
            return response.json()
        except Exception as exc:
            raise APIClientError("Malformed JSON response from login endpoint.") from exc

    def register(self, username: str, email: str, password: str) -> dict[str, Any]:
        """Register a new user account."""
        clean_user = username.strip() if isinstance(username, str) else ""
        clean_email = email.strip() if isinstance(email, str) else ""
        if not clean_user or not clean_email or not password:
            raise ValidationError("Username, email, and password cannot be empty.")

        url = f"{self.base_url}/auth/register"
        payload = {"username": clean_user, "email": clean_email, "password": password}

        try:
            if self._external_client is not None:
                response = self._external_client.post(url, json=payload)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise BackendUnavailableError("Registration request timed out. Please try again.") from exc
        except (httpx.ConnectError, httpx.RequestError) as exc:
            raise BackendUnavailableError(
                f"Unable to connect to backend server at {self.base_url}."
            ) from exc

        if response.status_code != 201:
            self._handle_response_error(response, "Registration")

        try:
            return response.json()
        except Exception as exc:
            raise APIClientError("Malformed JSON response from registration endpoint.") from exc

    def get_me(self, token: str) -> dict[str, Any]:
        """Retrieve profile information for currently authenticated user."""
        if not token or not isinstance(token, str):
            raise AuthenticationError("Authentication token is required.")

        url = f"{self.base_url}/auth/me"
        headers = {"Authorization": f"Bearer {token.strip()}"}

        try:
            if self._external_client is not None:
                response = self._external_client.get(url, headers=headers)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise BackendUnavailableError("Profile request timed out.") from exc
        except (httpx.ConnectError, httpx.RequestError) as exc:
            raise BackendUnavailableError(
                f"Unable to connect to backend server at {self.base_url}."
            ) from exc

        if response.status_code != 200:
            self._handle_response_error(response, "User profile lookup")

        try:
            return response.json()
        except Exception as exc:
            raise APIClientError("Malformed JSON response from user profile endpoint.") from exc

    def send_chat(self, token: str, session_id: str, message: str) -> dict[str, str]:
        """Submit a query to the regulatory chat agent."""
        if not token or not isinstance(token, str):
            raise AuthenticationError("Authentication token is required for chat.")
        clean_session = session_id.strip() if isinstance(session_id, str) else ""
        clean_msg = message.strip() if isinstance(message, str) else ""
        if not clean_session:
            raise ValidationError("session_id cannot be empty.")
        if not clean_msg:
            raise ValidationError("message cannot be empty.")

        url = f"{self.base_url}/api/v1/chat"
        headers = {
            "Authorization": f"Bearer {token.strip()}",
            "Content-Type": "application/json",
        }
        payload = {"session_id": clean_session, "message": clean_msg}

        try:
            if self._external_client is not None:
                response = self._external_client.post(url, json=payload, headers=headers)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise BackendUnavailableError("Chat request timed out waiting for assistant response.") from exc
        except (httpx.ConnectError, httpx.RequestError) as exc:
            raise BackendUnavailableError(
                f"Unable to connect to backend server at {self.base_url}."
            ) from exc

        if response.status_code != 200:
            self._handle_response_error(response, "Chat submission")

        try:
            return response.json()
        except Exception as exc:
            raise APIClientError("Malformed JSON response from chat endpoint.") from exc

    def clear_chat(self, token: str, session_id: str) -> dict[str, str]:
        """Clear active conversation memory for a session on the backend."""
        if not token or not isinstance(token, str):
            raise AuthenticationError("Authentication token is required.")
        clean_session = session_id.strip() if isinstance(session_id, str) else ""
        if not clean_session:
            raise ValidationError("session_id cannot be empty.")

        url = f"{self.base_url}/api/v1/chat/{clean_session}"
        headers = {"Authorization": f"Bearer {token.strip()}"}

        try:
            if self._external_client is not None:
                response = self._external_client.delete(url, headers=headers)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.delete(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise BackendUnavailableError("Clear chat request timed out.") from exc
        except (httpx.ConnectError, httpx.RequestError) as exc:
            raise BackendUnavailableError(
                f"Unable to connect to backend server at {self.base_url}."
            ) from exc

        if response.status_code != 200:
            self._handle_response_error(response, "Clear conversation")

        try:
            return response.json()
        except Exception as exc:
            raise APIClientError("Malformed JSON response from clear chat endpoint.") from exc
