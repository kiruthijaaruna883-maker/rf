"""Unit tests for the frontend session state management module."""

import unittest
from unittest.mock import patch

from frontend.session import (
    add_chat_message,
    clear_current_messages,
    generate_session_id,
    get_access_token,
    get_current_session_id,
    get_current_user,
    get_messages,
    init_session_state,
    is_authenticated,
    logout,
    set_authenticated_user,
    start_new_session,
)


class TestFrontendSessionState(unittest.TestCase):
    """Test suite verifying Streamlit session state manipulation and isolation."""

    def setUp(self) -> None:
        # Provide an isolated dictionary simulating st.session_state
        self.fake_session_state = {}
        self.patcher = patch("frontend.session.st.session_state", self.fake_session_state)
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()

    def test_init_session_state_sets_defaults(self) -> None:
        init_session_state()

        self.assertFalse(self.fake_session_state["authenticated"])
        self.assertIsNone(self.fake_session_state["access_token"])
        self.assertIsNone(self.fake_session_state["user_info"])
        self.assertIsInstance(self.fake_session_state["session_id"], str)
        self.assertEqual(len(self.fake_session_state["session_id"]), 32)
        self.assertEqual(self.fake_session_state["messages"], [])

    def test_set_authenticated_user(self) -> None:
        init_session_state()
        user_data = {"id": 1, "username": "reg_officer", "email": "officer@pharma.com"}

        set_authenticated_user("test-jwt-token-abc", user_data)

        self.assertTrue(is_authenticated())
        self.assertEqual(get_access_token(), "test-jwt-token-abc")
        self.assertEqual(get_current_user(), user_data)

    def test_logout_resets_auth_state_and_generates_new_session(self) -> None:
        init_session_state()
        set_authenticated_user("token", {"username": "user"})
        add_chat_message("user", "Confidential query")
        old_session_id = get_current_session_id()

        logout()

        self.assertFalse(is_authenticated())
        self.assertIsNone(get_access_token())
        self.assertIsNone(get_current_user())
        self.assertEqual(get_messages(), [])
        self.assertNotEqual(get_current_session_id(), old_session_id)

    def test_start_new_session_isolation(self) -> None:
        init_session_state()
        add_chat_message("user", "First session query")
        first_session_id = get_current_session_id()

        second_session_id = start_new_session()

        self.assertNotEqual(first_session_id, second_session_id)
        self.assertEqual(get_current_session_id(), second_session_id)
        self.assertEqual(get_messages(), [])

    def test_clear_current_messages_preserves_session_id(self) -> None:
        init_session_state()
        add_chat_message("user", "Query to be cleared")
        session_id = get_current_session_id()

        clear_current_messages()

        self.assertEqual(get_messages(), [])
        self.assertEqual(get_current_session_id(), session_id)

    def test_add_chat_message_validation(self) -> None:
        init_session_state()

        add_chat_message("user", "What is ICH Q1A?")
        add_chat_message("assistant", "ICH Q1A covers stability testing.")

        messages = get_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "user", "content": "What is ICH Q1A?"})
        self.assertEqual(messages[1], {"role": "assistant", "content": "ICH Q1A covers stability testing."})

        # Invalid roles
        with self.assertRaises(ValueError):
            add_chat_message("system", "Not allowed")
        with self.assertRaises(ValueError):
            add_chat_message("admin", "Not allowed")

        # Empty content
        with self.assertRaises(ValueError):
            add_chat_message("user", "")
        with self.assertRaises(ValueError):
            add_chat_message("user", "   ")

    def test_session_id_generation_is_random_and_unique(self) -> None:
        id1 = generate_session_id()
        id2 = generate_session_id()
        self.assertNotEqual(id1, id2)
        self.assertEqual(len(id1), 32)


if __name__ == "__main__":
    unittest.main()
