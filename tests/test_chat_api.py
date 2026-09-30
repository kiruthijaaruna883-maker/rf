"""Unit tests for FastAPI chat endpoint (POST /api/v1/chat)."""

import unittest
from unittest.mock import MagicMock, patch

from fastapi import status
from fastapi.testclient import TestClient

from app.agent.graph import (
    AgentExecutionError,
    AgentError,
    LLMAPIError,
    LLMConfigError,
    LLMError,
)
from app.api.auth import get_current_user
from app.database.models import ChatLog, User
from app.database.session import get_db
from app.main import app
from app.memory.redis_memory import RedisMemoryError


class TestChatApi(unittest.TestCase):
    """Test suite for POST /api/v1/chat."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.test_user = User(
            id=42,
            username="regulatory_officer",
            email="officer@pharma.com",
            hashed_password="hashed_secret",
            is_active=True,
        )
        self.mock_db = MagicMock()
        # Clean overrides before each test
        app.dependency_overrides = {}

    def tearDown(self) -> None:
        app.dependency_overrides = {}

    def _set_authenticated_user(self, user: User | None = None) -> None:
        active_user = user or self.test_user
        app.dependency_overrides[get_current_user] = lambda: active_user
        app.dependency_overrides[get_db] = lambda: self.mock_db

    # 1. Authentication
    def test_unauthenticated_request_returns_401(self) -> None:
        # No dependency override; unauthenticated request
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-1", "message": "Hello"},
        )
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    @patch("app.api.chat.run_agent")
    def test_authenticated_request_succeeds(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.return_value = "Regulatory guidance answer."

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-auth", "message": "What are the rules?"},
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.json(),
            {"session_id": "sess-auth", "response": "Regulatory guidance answer."},
        )

    # 2. Request validation
    def test_missing_session_id_returns_422(self) -> None:
        self._set_authenticated_user()
        response = self.client.post(
            "/api/v1/chat",
            json={"message": "Valid query"},
        )
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)

    def test_empty_session_id_returns_422(self) -> None:
        self._set_authenticated_user()
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "", "message": "Valid query"},
        )
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)

    def test_whitespace_only_session_id_returns_422(self) -> None:
        self._set_authenticated_user()
        for ws in ["   ", "\t", "\n  \t"]:
            with self.subTest(session_id=ws):
                response = self.client.post(
                    "/api/v1/chat",
                    json={"session_id": ws, "message": "Valid query"},
                )
                self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)

    def test_missing_message_returns_422(self) -> None:
        self._set_authenticated_user()
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-valid"},
        )
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)

    def test_empty_message_returns_422(self) -> None:
        self._set_authenticated_user()
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-valid", "message": ""},
        )
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)

    def test_whitespace_only_message_returns_422(self) -> None:
        self._set_authenticated_user()
        for ws in ["   ", "\t", "\n\r "]:
            with self.subTest(message=ws):
                response = self.client.post(
                    "/api/v1/chat",
                    json={"session_id": "sess-valid", "message": ws},
                )
                self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)

    @patch("app.api.chat.run_agent")
    def test_query_alias_field_accepted(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.return_value = "Answer via query alias."

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-alias", "query": "Query text here"},
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_run_agent.assert_called_once_with(
            session_id="sess-alias",
            query="Query text here",
            user_id="42",
        )

    # 3. Agent invocation and identity enforcement
    @patch("app.api.chat.run_agent")
    def test_agent_invoked_with_authenticated_user_id(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.return_value = "Response content"

        self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-100", "message": "Regulatory question"},
        )

        mock_run_agent.assert_called_once_with(
            session_id="sess-100",
            query="Regulatory question",
            user_id="42",
        )

    @patch("app.api.chat.run_agent")
    def test_client_supplied_user_id_is_not_authoritative(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.return_value = "Response content"

        # Attempt to spoof user identity via payload
        self.client.post(
            "/api/v1/chat",
            json={
                "session_id": "sess-spoof",
                "message": "Regulatory question",
                "user_id": "99999",
            },
        )

        # Verified: Authoritative authenticated user ID "42" is passed, NOT "99999"
        mock_run_agent.assert_called_once_with(
            session_id="sess-spoof",
            query="Regulatory question",
            user_id="42",
        )

    # 4. Session isolation at API level
    @patch("app.api.chat.run_agent")
    def test_session_id_passed_unchanged(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.return_value = "Isolated response"

        self.client.post(
            "/api/v1/chat",
            json={"session_id": "custom-session-uuid-789", "message": "Question"},
        )

        mock_run_agent.assert_called_once_with(
            session_id="custom-session-uuid-789",
            query="Question",
            user_id="42",
        )

    # 5. Response schema & security boundaries
    @patch("app.api.chat.run_agent")
    def test_response_contains_no_sensitive_secrets(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.return_value = "Safe public guidance."

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-safe", "message": "Verify stability requirements"},
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()

        self.assertIn("session_id", data)
        self.assertIn("response", data)
        # Check security boundaries: no tokens, passwords, keys, or credentials
        self.assertNotIn("password", str(data).lower())
        self.assertNotIn("token", str(data).lower())
        self.assertNotIn("secret", str(data).lower())
        self.assertNotIn("api_key", str(data).lower())

    # 6. Error handling
    @patch("app.api.chat.run_agent")
    def test_llm_config_error_returns_503(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.side_effect = LLMConfigError("OpenAI API key is a placeholder value")

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-err", "message": "Question"},
        )
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        # Detail is sanitized; internal message not leaked
        self.assertEqual(response.json()["detail"], "LLM service is not properly configured.")
        self.assertNotIn("OpenAI", response.json()["detail"])

    @patch("app.api.chat.run_agent")
    def test_llm_api_rate_limit_returns_429(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.side_effect = LLMAPIError("Rate limit exceeded", status_code=429)

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-err", "message": "Question"},
        )
        self.assertEqual(response.status_code, status.HTTP_429_TOO_MANY_REQUESTS)
        self.assertIn("rate limit", response.json()["detail"].lower())

    @patch("app.api.chat.run_agent")
    def test_llm_api_error_returns_502(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.side_effect = LLMAPIError("Bad gateway from provider", status_code=502)

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-err", "message": "Question"},
        )
        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)

    @patch("app.api.chat.run_agent")
    def test_llm_error_returns_502(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.side_effect = LLMError("Provider timeout")

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-err", "message": "Question"},
        )
        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)

    @patch("app.api.chat.run_agent")
    def test_redis_memory_error_returns_503(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.side_effect = RedisMemoryError("Redis connection refused")

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-err", "message": "Question"},
        )
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertNotIn("Redis", response.json()["detail"])
        self.assertIn("Conversation memory service is temporarily unavailable.", response.json()["detail"])

    @patch("app.api.chat.run_agent")
    def test_agent_execution_error_returns_500(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.side_effect = AgentExecutionError("Tool execution failed")

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-err", "message": "Question"},
        )
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    # 7. Audit log persistence
    @patch("app.api.chat.run_agent")
    def test_chat_log_audit_record_created(self, mock_run_agent: MagicMock) -> None:
        self._set_authenticated_user()
        mock_run_agent.return_value = "Logged assistant response."

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-audit", "message": "Audit this question"},
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.mock_db.add.assert_called_once()
        saved_log = self.mock_db.add.call_args[0][0]
        self.assertIsInstance(saved_log, ChatLog)
        self.assertEqual(saved_log.user_id, 42)
        self.assertEqual(saved_log.session_id, "sess-audit")
        self.assertEqual(saved_log.user_query, "Audit this question")
        self.assertEqual(saved_log.assistant_response, "Logged assistant response.")
        self.mock_db.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
