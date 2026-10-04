"""Unit tests for LangGraph agent orchestration layer."""

import unittest
from typing import get_type_hints
from unittest.mock import MagicMock, patch

from app.agent.graph import (
    AgentExecutionError,
    AgentError,
    LLMAPIError,
    LLMConfigError,
    LLMError,
    agent_graph,
    call_llm,
    create_agent_graph,
    extract_drug_name,
    route_intent,
    run_agent,
)
from app.agent.state import AgentState
from app.memory.redis_memory import ConversationMessage, RedisConversationMemory, RedisMemoryError


class TestAgentState(unittest.TestCase):
    """Test state definitions and validation."""

    def test_state_required_fields(self) -> None:
        hints = get_type_hints(AgentState)
        required_keys = {"session_id", "user_id", "messages", "query", "response"}
        for key in required_keys:
            self.assertIn(key, hints, f"Missing required state key: {key}")

    def test_valid_state_creation(self) -> None:
        msg = ConversationMessage(role="user", content="Prior query")
        state: AgentState = {
            "session_id": "test-session-1",
            "user_id": "user-42",
            "messages": [msg],
            "query": "What is the formulation?",
            "tool_name": "rag_search",
            "tool_input": "What is the formulation?",
            "tool_output": "Section 3.1: formulation details",
            "response": "Here is the formulation...",
        }
        self.assertEqual(state["session_id"], "test-session-1")
        self.assertEqual(len(state["messages"]), 1)
        self.assertEqual(state["tool_name"], "rag_search")


class TestAgentGraphConstruction(unittest.TestCase):
    """Test graph structure, nodes, and safety boundaries."""

    def test_graph_compiles_successfully(self) -> None:
        graph = create_agent_graph()
        self.assertIsNotNone(graph)
        self.assertTrue(hasattr(graph, "invoke"))

    def test_expected_nodes_exist(self) -> None:
        expected_nodes = {"load_memory", "agent", "execute_tool", "response"}
        for node in expected_nodes:
            self.assertIn(node, agent_graph.nodes, f"Expected node '{node}' missing from graph")

    def test_safety_boundary_no_regulatory_decision_nodes(self) -> None:
        forbidden_substrings = ["approve", "approval", "reject", "decision", "authorize"]
        for node_name in agent_graph.nodes:
            for forbidden in forbidden_substrings:
                self.assertNotIn(
                    forbidden,
                    node_name.lower(),
                    f"Graph contains prohibited autonomous decision node: {node_name}",
                )


class TestAgentRouting(unittest.TestCase):
    """Test deterministic intent routing to existing direct tools."""

    def test_route_regulatory_document_query_to_rag_search(self) -> None:
        queries = [
            "What does our uploaded stability document say?",
            "Find the section discussing stability conditions.",
            "What are the requirements under 21 CFR Part 312?",
            "Show internal SOP for document verification",
        ]
        for q in queries:
            with self.subTest(query=q):
                tool_name, tool_input = route_intent(q)
                self.assertEqual(tool_name, "rag_search")
                self.assertEqual(tool_input, q)

    def test_route_drug_query_to_drug_lookup(self) -> None:
        cases = [
            ("Look up the regulatory label information for aspirin.", "aspirin"),
            ("What does openFDA list for Advil?", "Advil"),
            ("Look up aspirin in openfda", "aspirin"),
            ("Drug lookup for Ibuprofen", "Ibuprofen"),
            ("What are the warnings for aspirin according to FDA labeling?", "aspirin"),
            ("What are the adverse reactions for aspirin according to FDA labeling?", "aspirin"),
        ]
        for query, expected_drug in cases:
            with self.subTest(query=query):
                tool_name, tool_input = route_intent(query)
                self.assertEqual(tool_name, "drug_lookup")
                self.assertEqual(tool_input, expected_drug)

    def test_route_user_lookup(self) -> None:
        # Case with explicit user_id passed
        tool_name, tool_input = route_intent("Look up user profile in user registry", user_id="42")
        self.assertEqual(tool_name, "user_lookup")
        self.assertEqual(tool_input, "42")

        # Case with user ID in query text
        tool_name, tool_input = route_intent("Look up user 99 permissions")
        self.assertEqual(tool_name, "user_lookup")
        self.assertEqual(tool_input, "99")

    def test_route_external_regulatory_search_to_web_search(self) -> None:
        queries = [
            "Check external regulatory web search for recent FDA announcements",
            "Perform a web search for agency announcements",
            "external search on recent EMA guidelines",
        ]
        for q in queries:
            with self.subTest(query=q):
                tool_name, tool_input = route_intent(q)
                self.assertEqual(tool_name, "web_search")
                self.assertEqual(tool_input, q)

    def test_route_conversational_direct_reply(self) -> None:
        greetings = ["Hello", "Hi", "help", "who are you", "what can you do"]
        for g in greetings:
            with self.subTest(greeting=g):
                tool_name, tool_input = route_intent(g)
                self.assertIsNone(tool_name)
                self.assertIsNone(tool_input)


class TestAgentExecutionAndTools(unittest.TestCase):
    """Test tool execution and response synthesis flows."""

    def setUp(self) -> None:
        self.mock_memory = MagicMock(spec=RedisConversationMemory)
        self.mock_memory.get_messages.return_value = []

    @patch("app.agent.graph.call_llm")
    @patch("app.agent.graph.rag_search")
    def test_rag_search_execution_and_synthesis(
        self,
        mock_rag: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        mock_rag.return_value = "[Source 1: stability.pdf | Doc ID: 1 | Chunk: 0 | Similarity: 0.95]\nStore at 25C."
        mock_llm.return_value = "Based on Source 1, the stability condition is 25C."

        response = run_agent(
            session_id="sess-rag",
            query="What does our uploaded stability document say?",
            memory=self.mock_memory,
        )

        mock_rag.assert_called_once_with("What does our uploaded stability document say?")
        mock_llm.assert_called_once()
        self.assertEqual(response, "Based on Source 1, the stability condition is 25C.")

    @patch("app.agent.graph.rag_search")
    def test_rag_search_no_evidence_returns_notice_without_llm(self, mock_rag: MagicMock) -> None:
        mock_rag.return_value = "No relevant regulatory guidance sections were found matching your query: 'missing topic'."

        response = run_agent(
            session_id="sess-no-rag",
            query="Search for missing topic in documents",
            memory=self.mock_memory,
        )
        self.assertIn("No relevant regulatory guidance sections were found", response)
        self.assertIn("Notice: No grounded internal guidance matched", response)

    @patch("app.agent.graph.drug_lookup")
    def test_drug_lookup_execution_formats_response_with_disclaimer(
        self,
        mock_drug: MagicMock,
    ) -> None:
        mock_drug.return_value = {
            "status": "found",
            "source": "openFDA",
            "drug_name": "aspirin",
            "brand_name": "Aspirin",
            "generic_name": "acetylsalicylic acid",
            "active_ingredients": ["Aspirin"],
            "manufacturer": "Bayer",
            "product_type": "HUMAN OTC DRUG",
            "indications_and_usage": "Pain relief",
            "warnings": "Reye's syndrome warning",
            "dosage_and_administration": "Take 1 tablet every 4 hours",
            "disclaimer": "This information is retrieved from official openFDA records. It does not constitute medical advice.",
        }

        response = run_agent(
            session_id="sess-drug",
            query="Look up the regulatory label information for aspirin.",
            memory=self.mock_memory,
        )

        mock_drug.assert_called_once_with("aspirin")
        self.assertIn("Aspirin", response)
        self.assertIn("acetylsalicylic acid", response)
        self.assertIn("Pain relief", response)
        self.assertIn("Medical Disclaimer:", response)

    @patch("app.agent.graph.user_lookup")
    def test_user_lookup_execution_formats_safe_metadata(self, mock_user: MagicMock) -> None:
        mock_user.return_value = {
            "found": True,
            "user_id": 42,
            "username": "regulatory_lead",
            "email": "lead@pharma.com",
            "is_active": True,
            "created_at": "2026-01-15T10:00:00",
        }

        response = run_agent(
            session_id="sess-user",
            query="Look up user profile in user registry",
            user_id="42",
            memory=self.mock_memory,
        )

        mock_user.assert_called_once_with("42")
        self.assertIn("User Registry Profile (User ID: 42):", response)
        self.assertIn("regulatory_lead", response)
        self.assertIn("lead@pharma.com", response)
        self.assertNotIn("password", response.lower())

    @patch("app.agent.graph.web_search")
    def test_web_search_preserves_unconfigured_message(self, mock_web: MagicMock) -> None:
        deferred_msg = (
            "Web search capability is currently not configured. "
            "Real-time external regulatory portal search will be enabled in a future release."
        )
        mock_web.return_value = deferred_msg

        response = run_agent(
            session_id="sess-web",
            query="Check external regulatory web search for recent FDA announcements",
            memory=self.mock_memory,
        )

        mock_web.assert_called_once_with("Check external regulatory web search for recent FDA announcements")
        self.assertEqual(response, deferred_msg)


class TestAgentMemoryIntegration(unittest.TestCase):
    """Test Redis conversation memory integration and session isolation."""

    @patch("app.agent.graph.call_llm")
    def test_previous_conversation_loaded_and_new_messages_saved(
        self,
        mock_llm: MagicMock,
    ) -> None:
        mock_llm.return_value = "I am ready to assist with your regulatory query."

        mock_memory = MagicMock(spec=RedisConversationMemory)
        prior_msgs = [
            ConversationMessage(role="user", content="Hi"),
            ConversationMessage(role="assistant", content="Hello! How can I help?"),
        ]
        mock_memory.get_messages.return_value = prior_msgs

        response = run_agent(
            session_id="session-mem-1",
            query="Hello again",
            memory=mock_memory,
        )

        # 1. Previous conversation was loaded
        mock_memory.get_messages.assert_called_once_with("session-mem-1")

        # 2. User message was stored
        mock_memory.add_message.assert_any_call("session-mem-1", "user", "Hello again")

        # 3. Assistant response was stored
        mock_memory.add_message.assert_any_call("session-mem-1", "assistant", response)

    @patch("app.agent.graph.web_search")
    def test_session_isolation_independent_histories(self, mock_web: MagicMock) -> None:
        mock_web.return_value = "Web search deferred."

        session_store: dict[str, list[ConversationMessage]] = {}

        class FakeMemory(RedisConversationMemory):
            def __init__(self) -> None:
                pass

            def get_messages(self, session_id: str) -> list[ConversationMessage]:
                return list(session_store.get(session_id, []))

            def add_message(self, session_id: str, role: str, content: str) -> None:
                session_store.setdefault(session_id, []).append(ConversationMessage(role=role, content=content))

        fake_mem = FakeMemory()

        run_agent(
            session_id="session-A",
            query="Check external regulatory web search for recent FDA announcements",
            memory=fake_mem,
        )
        run_agent(
            session_id="session-B",
            query="Check external regulatory web search for recent FDA announcements",
            memory=fake_mem,
        )

        history_a = fake_mem.get_messages("session-A")
        history_b = fake_mem.get_messages("session-B")

        self.assertEqual(len(history_a), 2)
        self.assertEqual(len(history_b), 2)
        self.assertEqual(session_store["session-A"], history_a)
        self.assertEqual(session_store["session-B"], history_b)


class TestAgentErrors(unittest.TestCase):
    """Test error handling in graph execution, validation, tools, memory, and LLM."""

    def test_empty_query_raises_value_error(self) -> None:
        for bad_query in ["", "   ", "\t\n", None]:
            with self.subTest(query=bad_query):
                with self.assertRaises(ValueError):
                    run_agent(session_id="valid-session", query=bad_query)

    def test_invalid_session_id_raises_value_error(self) -> None:
        for bad_session in ["", "   ", "\t\n", None]:
            with self.subTest(session=bad_session):
                with self.assertRaises(ValueError):
                    run_agent(session_id=bad_session, query="Valid query")

    def test_redis_failure_raises_clean_error(self) -> None:
        mock_memory = MagicMock(spec=RedisConversationMemory)
        mock_memory.get_messages.side_effect = RedisMemoryError("Redis connection refused")

        with self.assertRaises(RedisMemoryError):
            run_agent(
                session_id="fail-session",
                query="Look up the regulatory label information for aspirin.",
                memory=mock_memory,
            )

    @patch("app.agent.graph.rag_search")
    def test_tool_failure_raises_agent_execution_error(self, mock_rag: MagicMock) -> None:
        mock_rag.side_effect = RuntimeError("Database connection pool exhausted")
        mock_memory = MagicMock(spec=RedisConversationMemory)
        mock_memory.get_messages.return_value = []

        with self.assertRaises(AgentExecutionError):
            run_agent(
                session_id="sess-err",
                query="What does our uploaded stability document say?",
                memory=mock_memory,
            )

    @patch("app.agent.graph.call_llm")
    @patch("app.agent.graph.rag_search")
    def test_llm_failure_raises_llm_error(
        self,
        mock_rag: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        mock_rag.return_value = "[Source 1: doc.pdf | Doc ID: 1 | Chunk: 0 | Similarity: 0.9]\nEvidence."
        mock_llm.side_effect = LLMAPIError("Rate limit exceeded", status_code=429)

        mock_memory = MagicMock(spec=RedisConversationMemory)
        mock_memory.get_messages.return_value = []

        with self.assertRaises(LLMError):
            run_agent(
                session_id="sess-llm-fail",
                query="What does our uploaded stability document say?",
                memory=mock_memory,
            )

    def test_call_llm_unconfigured_or_placeholder_key_raises_config_error(self) -> None:
        with self.assertRaises(LLMConfigError):
            call_llm(
                messages=[{"role": "user", "content": "hi"}],
                api_key="your-openai-api-key-here",
            )
        with self.assertRaises(LLMConfigError):
            call_llm(
                messages=[{"role": "user", "content": "hi"}],
                api_key="",
            )


if __name__ == "__main__":
    unittest.main()
