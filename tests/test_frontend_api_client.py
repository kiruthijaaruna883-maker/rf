"""Unit tests for the Streamlit frontend API client."""

from unittest.mock import MagicMock
import unittest
import httpx

from frontend.api_client import (
    APIClientError,
    AuthenticationError,
    BackendUnavailableError,
    ConflictError,
    ForbiddenError,
    RateLimitError,
    RegulatoryChatClient,
    ServerError,
    ValidationError,
)


class TestRegulatoryChatClient(unittest.TestCase):
    """Test suite for RegulatoryChatClient HTTP interactions and error translation."""

    def setUp(self) -> None:
        self.mock_http = MagicMock(spec=httpx.Client)
        self.client = RegulatoryChatClient(
            base_url="http://testserver:8000",
            timeout=10.0,
            client=self.mock_http,
        )

    # --- Login Tests ---

    def test_login_success(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "fake-jwt-token-123", "token_type": "bearer"}
        self.mock_http.post.return_value = mock_resp

        result = self.client.login("regulatory_officer", "SecretPass123!")

        self.mock_http.post.assert_called_once_with(
            "http://testserver:8000/auth/login",
            data={"username": "regulatory_officer", "password": "SecretPass123!"},
        )
        self.assertEqual(result["access_token"], "fake-jwt-token-123")
        self.assertEqual(result["token_type"], "bearer")

    def test_login_invalid_credentials_raises_auth_error(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"detail": "Incorrect username or password"}
        self.mock_http.post.return_value = mock_resp

        with self.assertRaises(AuthenticationError) as ctx:
            self.client.login("officer", "wrongpass")
        self.assertIn("Incorrect username or password", ctx.exception.message)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_login_forbidden_inactive_user(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 403
        mock_resp.json.return_value = {"detail": "Inactive user account"}
        self.mock_http.post.return_value = mock_resp

        with self.assertRaises(ForbiddenError) as ctx:
            self.client.login("inactive_user", "pass")
        self.assertEqual(ctx.exception.status_code, 403)

    def test_login_empty_inputs_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            self.client.login("", "pass")
        with self.assertRaises(ValidationError):
            self.client.login("user", "")

    def test_login_timeout_raises_backend_unavailable(self) -> None:
        self.mock_http.post.side_effect = httpx.TimeoutException("Read timeout")
        with self.assertRaises(BackendUnavailableError):
            self.client.login("user", "pass")

    def test_login_connect_error_raises_backend_unavailable(self) -> None:
        self.mock_http.post.side_effect = httpx.ConnectError("Connection refused")
        with self.assertRaises(BackendUnavailableError):
            self.client.login("user", "pass")

    def test_login_malformed_json_raises_api_client_error(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        self.mock_http.post.return_value = mock_resp

        with self.assertRaises(APIClientError) as ctx:
            self.client.login("user", "pass")
        self.assertIn("Malformed JSON", ctx.exception.message)

    # --- Register Tests ---

    def test_register_success(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "id": 10,
            "username": "new_reg_user",
            "email": "user@pharma.com",
            "is_active": True,
        }
        self.mock_http.post.return_value = mock_resp

        result = self.client.register("new_reg_user", "user@pharma.com", "Password123!")

        self.mock_http.post.assert_called_once_with(
            "http://testserver:8000/auth/register",
            json={"username": "new_reg_user", "email": "user@pharma.com", "password": "Password123!"},
        )
        self.assertEqual(result["id"], 10)
        self.assertEqual(result["username"], "new_reg_user")

    def test_register_conflict_raises_conflict_error(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 409
        mock_resp.json.return_value = {"detail": "Username already registered"}
        self.mock_http.post.return_value = mock_resp

        with self.assertRaises(ConflictError) as ctx:
            self.client.register("existing_user", "user@pharma.com", "pass")
        self.assertIn("already registered", ctx.exception.message)

    def test_register_validation_error_raises_validation_error(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 422
        mock_resp.json.return_value = {"detail": [{"loc": ["body", "email"], "msg": "value is not a valid email"}]}
        self.mock_http.post.return_value = mock_resp

        with self.assertRaises(ValidationError) as ctx:
            self.client.register("user", "bademail", "pass")
        self.assertIn("valid email", ctx.exception.message)

    def test_register_empty_fields_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            self.client.register("", "a@b.com", "pass")
        with self.assertRaises(ValidationError):
            self.client.register("user", "", "pass")
        with self.assertRaises(ValidationError):
            self.client.register("user", "a@b.com", "")

    # --- Get Profile (get_me) Tests ---

    def test_get_me_success(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": 42, "username": "lead_officer", "email": "lead@pharma.com", "is_active": True}
        self.mock_http.get.return_value = mock_resp

        result = self.client.get_me(token="valid-token-xyz")

        self.mock_http.get.assert_called_once_with(
            "http://testserver:8000/auth/me",
            headers={"Authorization": "Bearer valid-token-xyz"},
        )
        self.assertEqual(result["id"], 42)
        self.assertEqual(result["username"], "lead_officer")

    def test_get_me_unauthorized_raises_auth_error(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"detail": "Could not validate credentials"}
        self.mock_http.get.return_value = mock_resp

        with self.assertRaises(AuthenticationError):
            self.client.get_me(token="expired-token")

    def test_get_me_empty_token_raises_auth_error(self) -> None:
        with self.assertRaises(AuthenticationError):
            self.client.get_me(token="")

    # --- Chat (send_chat) Tests ---

    def test_send_chat_success(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "session_id": "sess-alpha-01",
            "response": "According to ICH Q1A(R2), accelerated stability testing requires 40C.",
        }
        self.mock_http.post.return_value = mock_resp

        result = self.client.send_chat(
            token="valid-token",
            session_id="sess-alpha-01",
            message="What are accelerated stability testing conditions?",
        )

        self.mock_http.post.assert_called_once_with(
            "http://testserver:8000/api/v1/chat",
            json={"session_id": "sess-alpha-01", "message": "What are accelerated stability testing conditions?"},
            headers={"Authorization": "Bearer valid-token", "Content-Type": "application/json"},
        )
        self.assertEqual(result["session_id"], "sess-alpha-01")
        self.assertIn("ICH Q1A(R2)", result["response"])

    def test_send_chat_empty_message_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            self.client.send_chat(token="token", session_id="sess-1", message="")
        with self.assertRaises(ValidationError):
            self.client.send_chat(token="token", session_id="sess-1", message="   	 ")

    def test_send_chat_empty_session_id_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            self.client.send_chat(token="token", session_id="", message="Query")

    def test_send_chat_empty_token_raises_auth_error(self) -> None:
        with self.assertRaises(AuthenticationError):
            self.client.send_chat(token="", session_id="sess-1", message="Query")

    def test_send_chat_rate_limit_raises_rate_limit_error(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"detail": "Rate limit exceeded"}
        self.mock_http.post.return_value = mock_resp

        with self.assertRaises(RateLimitError):
            self.client.send_chat(token="token", session_id="sess-1", message="Query")

    def test_send_chat_backend_unavailable_raises_backend_unavailable(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 503
        mock_resp.json.return_value = {"detail": "Conversation memory unavailable"}
        self.mock_http.post.return_value = mock_resp

        with self.assertRaises(BackendUnavailableError):
            self.client.send_chat(token="token", session_id="sess-1", message="Query")

    def test_send_chat_server_error_raises_server_error(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"detail": "Agent tool execution failed"}
        self.mock_http.post.return_value = mock_resp

        with self.assertRaises(ServerError):
            self.client.send_chat(token="token", session_id="sess-1", message="Query")

    def test_send_chat_timeout_raises_backend_unavailable(self) -> None:
        self.mock_http.post.side_effect = httpx.TimeoutException("Chat generation timeout")
        with self.assertRaises(BackendUnavailableError):
            self.client.send_chat(token="token", session_id="sess-1", message="Query")

    # --- Clear Chat (clear_chat) Tests ---

    def test_clear_chat_success(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "session_id": "sess-clear-01",
            "message": "Conversation history cleared successfully.",
        }
        self.mock_http.delete.return_value = mock_resp

        result = self.client.clear_chat(token="valid-token", session_id="sess-clear-01")

        self.mock_http.delete.assert_called_once_with(
            "http://testserver:8000/api/v1/chat/sess-clear-01",
            headers={"Authorization": "Bearer valid-token"},
        )
        self.assertEqual(result["session_id"], "sess-clear-01")

    def test_clear_chat_forbidden_raises_forbidden_error(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 403
        mock_resp.json.return_value = {"detail": "Not authorized to clear this chat session."}
        self.mock_http.delete.return_value = mock_resp

        with self.assertRaises(ForbiddenError):
            self.client.clear_chat(token="token", session_id="someone-elses-session")

    def test_clear_chat_empty_token_raises_auth_error(self) -> None:
        with self.assertRaises(AuthenticationError):
            self.client.clear_chat(token="", session_id="sess-1")

    # --- Upload Document Tests ---

    def test_upload_document_success(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "id": 1,
            "filename": "guideline.txt",
            "chunks_count": 3,
            "message": "Document successfully ingested into RAG vector store.",
        }
        self.mock_http.post.return_value = mock_resp

        result = self.client.upload_document(
            token="valid-token",
            filename="guideline.txt",
            file_bytes=b"Sample content",
            content_type="text/plain",
        )

        self.mock_http.post.assert_called_once()
        call_args, call_kwargs = self.mock_http.post.call_args
        self.assertEqual(call_args[0], "http://testserver:8000/documents/upload")
        self.assertEqual(call_kwargs["headers"], {"Authorization": "Bearer valid-token"})
        self.assertIn("file", call_kwargs["files"])
        self.assertEqual(result["chunks_count"], 3)
        self.assertIn("successfully ingested", result["message"])

    def test_upload_document_empty_token_raises_auth_error(self) -> None:
        with self.assertRaises(AuthenticationError):
            self.client.upload_document(
                token="",
                filename="doc.txt",
                file_bytes=b"content",
            )

    def test_upload_document_empty_filename_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            self.client.upload_document(
                token="token",
                filename="   ",
                file_bytes=b"content",
            )

    def test_upload_document_empty_bytes_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            self.client.upload_document(
                token="token",
                filename="doc.txt",
                file_bytes=b"",
            )

    def test_upload_document_timeout_raises_backend_unavailable(self) -> None:
        self.mock_http.post.side_effect = httpx.TimeoutException("Read timed out")
        with self.assertRaises(BackendUnavailableError):
            self.client.upload_document(
                token="token",
                filename="doc.txt",
                file_bytes=b"content",
            )

    def test_upload_document_error_status_translates(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 422
        mock_resp.json.return_value = {"detail": "Failed to extract text from document"}
        self.mock_http.post.return_value = mock_resp

        with self.assertRaises(APIClientError) as ctx:
            self.client.upload_document(
                token="token",
                filename="doc.pdf",
                file_bytes=b"corrupt",
            )
        self.assertIn("Failed to extract text", str(ctx.exception))

    def test_no_credentials_leaked_in_exception_messages(self) -> None:
        secret_pass = "UltraSecretPassword999!"
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"detail": "Incorrect username or password"}
        self.mock_http.post.return_value = mock_resp

        with self.assertRaises(AuthenticationError) as ctx:
            self.client.login("user", secret_pass)
        self.assertNotIn(secret_pass, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
