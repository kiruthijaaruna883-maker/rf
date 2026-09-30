"""Unit tests for Redis conversation memory."""

import json
import unittest
from unittest.mock import MagicMock, patch

import redis
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    RedisError,
    TimeoutError as RedisTimeoutError,
)

from app.memory.redis_memory import (
    ConversationMessage,
    RedisConversationMemory,
    RedisMemoryError,
)


class TestRedisConversationMemory(unittest.TestCase):
    """Test suite for RedisConversationMemory and ConversationMessage."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=redis.Redis)
        self.memory = RedisConversationMemory(client=self.mock_client)

    # 1. Initialization / configuration
    @patch("app.memory.redis_memory.redis.Redis")
    def test_initialization_uses_configured_settings(self, mock_redis_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_redis_cls.return_value = mock_instance

        mem = RedisConversationMemory()
        mock_redis_cls.assert_called_once()
        _, kwargs = mock_redis_cls.call_args
        self.assertEqual(kwargs.get("host"), "localhost")
        self.assertEqual(kwargs.get("port"), 6379)
        self.assertTrue(kwargs.get("decode_responses"))
        self.assertEqual(mem.key_prefix, "chat:memory")
        self.assertEqual(mem.max_messages, 50)
        self.assertEqual(mem.ttl_seconds, 86400)
        self.assertIs(mem.client, mock_instance)

    def test_custom_prefix_and_injected_client(self) -> None:
        mem = RedisConversationMemory(
            client=self.mock_client,
            key_prefix="custom:prefix",
            max_messages=10,
            ttl_seconds=3600,
        )
        self.assertEqual(mem.key_prefix, "custom:prefix")
        self.assertEqual(mem.max_messages, 10)
        self.assertEqual(mem.ttl_seconds, 3600)
        self.assertIs(mem.client, self.mock_client)

    # 2. Validation
    def test_validation_empty_or_whitespace_session_id(self) -> None:
        invalid_sessions = ["", "   ", "\t\n", None, 123]
        for invalid in invalid_sessions:
            with self.subTest(session=invalid):
                with self.assertRaises(ValueError):
                    self.memory.add_message(invalid, "user", "Hello")
                with self.assertRaises(ValueError):
                    self.memory.get_messages(invalid)
                with self.assertRaises(ValueError):
                    self.memory.clear(invalid)

    def test_validation_invalid_role(self) -> None:
        invalid_roles = ["system", "admin", "bot", "moderator", "", "USER", None]
        for role in invalid_roles:
            with self.subTest(role=role):
                with self.assertRaises(ValueError):
                    self.memory.add_message("session-1", role, "Hello")

    def test_validation_empty_or_whitespace_content(self) -> None:
        invalid_contents = ["", "   ", "\n\t", None, 123]
        for content in invalid_contents:
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    self.memory.add_message("session-1", "user", content)

    def test_conversation_message_direct_validation(self) -> None:
        msg = ConversationMessage(role="user", content="Valid message")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Valid message")

        with self.assertRaises(ValueError):
            ConversationMessage(role="system", content="Invalid role")
        with self.assertRaises(ValueError):
            ConversationMessage(role="user", content="")

    # 3. Add message
    def test_add_user_message_serialization_and_rpush(self) -> None:
        self.memory.add_message("sess-1", "user", "What are the rules?")

        expected_key = "chat:memory:sess-1"
        self.mock_client.rpush.assert_called_once()
        call_key, call_payload = self.mock_client.rpush.call_args[0]
        self.assertEqual(call_key, expected_key)

        data = json.loads(call_payload)
        self.assertEqual(data, {"role": "user", "content": "What are the rules?"})
        # Check security boundary: only role and content stored
        self.assertEqual(set(data.keys()), {"role", "content"})

        # Bounded history and TTL checks
        self.mock_client.ltrim.assert_called_once_with(expected_key, -50, -1)
        self.mock_client.expire.assert_called_once_with(expected_key, 86400)

    def test_add_assistant_message_serialization_and_rpush(self) -> None:
        self.memory.add_message("sess-2", "assistant", "According to 21 CFR...")

        expected_key = "chat:memory:sess-2"
        self.mock_client.rpush.assert_called_once()
        call_key, call_payload = self.mock_client.rpush.call_args[0]
        self.assertEqual(call_key, expected_key)

        data = json.loads(call_payload)
        self.assertEqual(data, {"role": "assistant", "content": "According to 21 CFR..."})

    # 4. Get messages
    def test_get_messages_empty_session_returns_empty_list(self) -> None:
        self.mock_client.lrange.return_value = []
        messages = self.memory.get_messages("empty-sess")
        self.mock_client.lrange.assert_called_once_with("chat:memory:empty-sess", 0, -1)
        self.assertEqual(messages, [])

    def test_get_messages_reconstructs_messages_in_chronological_order(self) -> None:
        raw_items = [
            json.dumps({"role": "user", "content": "First user message"}),
            json.dumps({"role": "assistant", "content": "First response"}),
            json.dumps({"role": "user", "content": "Second user question"}),
        ]
        self.mock_client.lrange.return_value = raw_items

        messages = self.memory.get_messages("sess-ordered")
        self.mock_client.lrange.assert_called_once_with("chat:memory:sess-ordered", 0, -1)
        self.assertEqual(len(messages), 3)

        self.assertEqual(messages[0], ConversationMessage(role="user", content="First user message"))
        self.assertEqual(messages[1], ConversationMessage(role="assistant", content="First response"))
        self.assertEqual(messages[2], ConversationMessage(role="user", content="Second user question"))

    def test_get_messages_malformed_json_raises_redis_memory_error(self) -> None:
        self.mock_client.lrange.return_value = ["not a valid json string"]
        with self.assertRaises(RedisMemoryError):
            self.memory.get_messages("sess-corrupt")

    def test_get_messages_malformed_structure_raises_redis_memory_error(self) -> None:
        bad_structures = [
            json.dumps({"role": "user"}),  # missing content
            json.dumps({"content": "hi"}),  # missing role
            json.dumps({"role": "unknown", "content": "hi"}),  # invalid role
            json.dumps(["not", "a", "dict"]),  # list instead of dict
        ]
        for bad in bad_structures:
            with self.subTest(bad=bad):
                self.mock_client.lrange.return_value = [bad]
                with self.assertRaises(RedisMemoryError):
                    self.memory.get_messages("sess-corrupt")

    # 5. Clear
    def test_clear_deletes_only_requested_session_key(self) -> None:
        self.memory.clear("sess-delete")
        self.mock_client.delete.assert_called_once_with("chat:memory:sess-delete")

    # 6. Session isolation
    def test_session_isolation(self) -> None:
        session_store: dict[str, list[str]] = {}

        def fake_rpush(key: str, val: str) -> None:
            session_store.setdefault(key, []).append(val)

        def fake_lrange(key: str, start: int, stop: int) -> list[str]:
            return session_store.get(key, [])

        fake_client = MagicMock(spec=redis.Redis)
        fake_client.rpush.side_effect = fake_rpush
        fake_client.lrange.side_effect = fake_lrange
        fake_client.ltrim.return_value = True
        fake_client.expire.return_value = True

        isolated_memory = RedisConversationMemory(client=fake_client)

        isolated_memory.add_message("session-A", "user", "Message for session A")
        isolated_memory.add_message("session-B", "user", "Message for session B")

        messages_a = isolated_memory.get_messages("session-A")
        messages_b = isolated_memory.get_messages("session-B")

        self.assertEqual(len(messages_a), 1)
        self.assertEqual(messages_a[0].content, "Message for session A")

        self.assertEqual(len(messages_b), 1)
        self.assertEqual(messages_b[0].content, "Message for session B")

        # Session A messages must never appear in Session B
        self.assertNotEqual(messages_a, messages_b)
        self.assertNotIn(messages_a[0], messages_b)

    # 7. History limit (bounded history)
    def test_history_limit_retains_newest_and_trims_oldest(self) -> None:
        session_store: dict[str, list[str]] = {}

        def fake_rpush(key: str, val: str) -> None:
            session_store.setdefault(key, []).append(val)

        def fake_ltrim(key: str, start: int, stop: int) -> None:
            if key in session_store:
                if stop == -1 and start < 0:
                    session_store[key] = session_store[key][start:]

        def fake_lrange(key: str, start: int, stop: int) -> list[str]:
            return session_store.get(key, [])

        fake_client = MagicMock(spec=redis.Redis)
        fake_client.rpush.side_effect = fake_rpush
        fake_client.ltrim.side_effect = fake_ltrim
        fake_client.lrange.side_effect = fake_lrange

        bounded_memory = RedisConversationMemory(client=fake_client, max_messages=3)

        for i in range(1, 6):
            bounded_memory.add_message("sess-limit", "user", f"Message {i}")

        messages = bounded_memory.get_messages("sess-limit")
        self.assertEqual(len(messages), 3)
        self.assertEqual([m.content for m in messages], ["Message 3", "Message 4", "Message 5"])

    # 8. Redis failure handling
    def test_redis_connection_error_on_add_message(self) -> None:
        self.mock_client.rpush.side_effect = RedisConnectionError("Redis connection refused")
        with self.assertRaises(RedisMemoryError):
            self.memory.add_message("sess-fail", "user", "Hello")

    def test_redis_timeout_error_on_get_messages(self) -> None:
        self.mock_client.lrange.side_effect = RedisTimeoutError("Redis timed out")
        with self.assertRaises(RedisMemoryError):
            self.memory.get_messages("sess-fail")

    def test_redis_error_on_clear(self) -> None:
        self.mock_client.delete.side_effect = RedisError("Redis cluster error")
        with self.assertRaises(RedisMemoryError):
            self.memory.clear("sess-fail")

    def test_redis_memory_error_subclasses_redis_error(self) -> None:
        self.assertTrue(issubclass(RedisMemoryError, RedisError))


if __name__ == "__main__":
    unittest.main()
