"""Integration test suite connecting RegulatoryChatClient to FastAPI backend."""

from unittest.mock import MagicMock, patch
import unittest
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.database.models import User
from app.database.session import get_db
from app.main import app
from frontend.api_client import RegulatoryChatClient


class TestFrontendBackendIntegration(unittest.TestCase):
    """End-to-end integration tests between frontend API client and FastAPI backend."""

    def setUp(self) -> None:
        self.test_client = TestClient(app)
        self.api_client = RegulatoryChatClient(
            base_url="http://testserver",
            client=self.test_client,
        )
        self.mock_db = MagicMock()
        app.dependency_overrides = {
            get_db: lambda: self.mock_db,
        }

    def tearDown(self) -> None:
        app.dependency_overrides = {}

    def test_client_against_fastapi_unauthorized_rejection(self) -> None:
        app.dependency_overrides = {get_db: lambda: self.mock_db}
        with self.assertRaises(Exception):
            self.api_client.send_chat(
                token="invalid-token",
                session_id="integration-sess-1",
                message="Regulatory question",
            )

    @patch("app.api.chat.run_agent")
    def test_client_against_fastapi_successful_chat_flow(self, mock_run_agent: MagicMock) -> None:
        mock_run_agent.return_value = "Grounded response: ICH Q1A accelerated condition is 40C."

        mock_user = MagicMock(spec=User)
        mock_user.id = 55
        mock_user.username = "qa_specialist"
        mock_user.is_active = True
        app.dependency_overrides[get_current_user] = lambda: mock_user

        resp = self.api_client.send_chat(
            token="valid-mock-jwt",
            session_id="integration-sess-55",
            message="What are stability conditions?",
        )

        self.assertEqual(resp["session_id"], "integration-sess-55")
        self.assertIn("ICH Q1A", resp["response"])

    @patch("app.api.chat.RedisConversationMemory")
    def test_client_against_fastapi_clear_chat_flow(self, mock_memory_cls: MagicMock) -> None:
        mock_mem = MagicMock()
        mock_memory_cls.return_value = mock_mem

        mock_user = MagicMock(spec=User)
        mock_user.id = 55
        mock_user.username = "qa_specialist"
        mock_user.is_active = True
        app.dependency_overrides[get_current_user] = lambda: mock_user

        # Ensure no existing logs to trigger cross-user check
        self.mock_db.query.return_value.filter.return_value.all.return_value = []

        resp = self.api_client.clear_chat(
            token="valid-mock-jwt",
            session_id="integration-sess-55",
        )

        self.assertEqual(resp["session_id"], "integration-sess-55")
        self.assertIn("cleared successfully", resp["message"].lower())


if __name__ == "__main__":
    unittest.main()
