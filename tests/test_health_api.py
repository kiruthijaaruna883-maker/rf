"""Unit and integration tests for the GET /health operational probe endpoint."""

from unittest.mock import MagicMock, patch
import unittest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.health import check_database, check_redis
from app.main import app


class TestHealthAPI(unittest.TestCase):
    """Test suite for the GET /health operational readiness endpoint."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_endpoint_exists_and_requires_no_auth(self) -> None:
        """Verify GET /health is accessible without Authorization headers."""
        with patch("app.api.health.check_database", return_value="healthy"),              patch("app.api.health.check_redis", return_value="healthy"):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_health_healthy_response_structure(self) -> None:
        """Verify response schema when all operational components are healthy."""
        with patch("app.api.health.check_database", return_value="healthy"),              patch("app.api.health.check_redis", return_value="healthy"):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            data = response.json()

            self.assertEqual(data["status"], "healthy")
            self.assertEqual(data["database"], "healthy")
            self.assertEqual(data["redis"], "healthy")

    def test_health_unhealthy_when_database_fails(self) -> None:
        """Verify 503 Service Unavailable returned when database connectivity fails."""
        with patch("app.api.health.check_database", return_value="unhealthy"),              patch("app.api.health.check_redis", return_value="healthy"):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
            data = response.json()

            self.assertEqual(data["status"], "unhealthy")
            self.assertEqual(data["database"], "unhealthy")
            self.assertEqual(data["redis"], "healthy")

    def test_health_unhealthy_when_redis_fails(self) -> None:
        """Verify 503 Service Unavailable returned when Redis connectivity fails."""
        with patch("app.api.health.check_database", return_value="healthy"),              patch("app.api.health.check_redis", return_value="unhealthy"):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
            data = response.json()

            self.assertEqual(data["status"], "unhealthy")
            self.assertEqual(data["database"], "healthy")
            self.assertEqual(data["redis"], "unhealthy")

    def test_health_unhealthy_when_both_dependencies_fail(self) -> None:
        """Verify 503 Service Unavailable returned when both dependencies fail."""
        with patch("app.api.health.check_database", return_value="unhealthy"),              patch("app.api.health.check_redis", return_value="unhealthy"):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
            data = response.json()

            self.assertEqual(data["status"], "unhealthy")
            self.assertEqual(data["database"], "unhealthy")
            self.assertEqual(data["redis"], "unhealthy")

    @patch("app.api.health.engine.connect")
    def test_check_database_function_success(self, mock_connect: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn

        result = check_database()
        self.assertEqual(result, "healthy")
        mock_conn.execute.assert_called_once()

    @patch("app.api.health.engine.connect")
    def test_check_database_function_failure(self, mock_connect: MagicMock) -> None:
        mock_connect.side_effect = RuntimeError("PostgreSQL connection refused")

        result = check_database()
        self.assertEqual(result, "unhealthy")

    @patch("app.api.health.redis.Redis")
    def test_check_redis_function_success(self, mock_redis_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis_cls.return_value = mock_client

        result = check_redis()
        self.assertEqual(result, "healthy")

    @patch("app.api.health.redis.Redis")
    def test_check_redis_function_failure(self, mock_redis_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.ping.side_effect = RuntimeError("Redis connection error")
        mock_redis_cls.return_value = mock_client

        result = check_redis()
        self.assertEqual(result, "unhealthy")


if __name__ == "__main__":
    unittest.main()
