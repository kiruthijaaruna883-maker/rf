"""Integration tests for the DELETE /api/v1/chat/{session_id} endpoint."""

from unittest.mock import MagicMock, patch
import unittest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.database.models import ChatLog, User
from app.database.session import get_db
from app.main import app
from app.memory.redis_memory import RedisMemoryError


class TestChatClearAPI(unittest.TestCase):
    """Test suite for the chat session memory clear endpoint."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.mock_user = MagicMock(spec=User)
        self.mock_user.id = 42
        self.mock_user.username = "regulatory_officer"
        self.mock_user.email = "officer@pharma.com"
        self.mock_user.is_active = True

        self.mock_db = MagicMock()
        app.dependency_overrides = {}

    def tearDown(self) -> None:
        app.dependency_overrides = {}

    def _set_authenticated_user(self, user: User) -> None:
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_db] = lambda: self.mock_db

    def test_clear_chat_unauthenticated_returns_401(self) -> None:
        app.dependency_overrides = {}
        response = self.client.delete("/api/v1/chat/test-session-123")
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_clear_chat_empty_session_id_returns_404_or_400(self) -> None:
        self._set_authenticated_user(self.mock_user)
        # Empty whitespace path parameter
        response = self.client.delete("/api/v1/chat/%20%20")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch("app.api.chat.RedisConversationMemory")
    def test_clear_chat_success_for_session_owner(self, mock_memory_cls: MagicMock) -> None:
        self._set_authenticated_user(self.mock_user)
        mock_mem_instance = MagicMock()
        mock_memory_cls.return_value = mock_mem_instance

        # Existing log belonging to this user
        existing_log = MagicMock(spec=ChatLog)
        existing_log.user_id = 42
        existing_log.session_id = "sess-owner-100"
        self.mock_db.query.return_value.filter.return_value.all.return_value = [existing_log]

        response = self.client.delete("/api/v1/chat/sess-owner-100")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["session_id"], "sess-owner-100")
        self.assertIn("cleared successfully", data["message"].lower())
        mock_mem_instance.clear.assert_called_once_with("sess-owner-100")

    @patch("app.api.chat.RedisConversationMemory")
    def test_clear_chat_cross_user_forbidden(self, mock_memory_cls: MagicMock) -> None:
        self._set_authenticated_user(self.mock_user)
        mock_mem_instance = MagicMock()
        mock_memory_cls.return_value = mock_mem_instance

        # Existing log belonging to a DIFFERENT user (user_id 999 != 42)
        other_user_log = MagicMock(spec=ChatLog)
        other_user_log.user_id = 999
        other_user_log.session_id = "sess-victim-200"
        self.mock_db.query.return_value.filter.return_value.all.return_value = [other_user_log]

        response = self.client.delete("/api/v1/chat/sess-victim-200")

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn("Not authorized", response.json()["detail"])
        mock_mem_instance.clear.assert_not_called()

    @patch("app.api.chat.RedisConversationMemory")
    def test_clear_chat_new_session_without_logs_succeeds(self, mock_memory_cls: MagicMock) -> None:
        self._set_authenticated_user(self.mock_user)
        mock_mem_instance = MagicMock()
        mock_memory_cls.return_value = mock_mem_instance

        # No existing logs for brand-new session
        self.mock_db.query.return_value.filter.return_value.all.return_value = []

        response = self.client.delete("/api/v1/chat/brand-new-sess")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_mem_instance.clear.assert_called_once_with("brand-new-sess")

    @patch("app.api.chat.RedisConversationMemory")
    def test_clear_chat_redis_failure_returns_503(self, mock_memory_cls: MagicMock) -> None:
        self._set_authenticated_user(self.mock_user)
        mock_mem_instance = MagicMock()
        mock_mem_instance.clear.side_effect = RedisMemoryError("Redis connection refused")
        mock_memory_cls.return_value = mock_mem_instance

        self.mock_db.query.return_value.filter.return_value.all.return_value = []

        response = self.client.delete("/api/v1/chat/sess-redis-fail")

        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertIn("temporarily unavailable", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
