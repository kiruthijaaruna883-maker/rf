"""End-to-end integration test suite for the Phase 5 agentic chat pipeline.

Tests the complete integrated flow across all Phase 5 layers:
FastAPI (/api/v1/chat) -> Authentication -> LangGraph Agent -> Direct Tools (RAG, drug lookup)
-> Redis Conversation Memory (multi-turn history) -> LLM Synthesis -> PostgreSQL ChatLog Persistence.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi import status
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.database.models import ChatLog, User
from app.database.session import SessionLocal
from app.main import app
from app.memory.redis_memory import RedisConversationMemory


class FakeRedis:
    """In-memory Redis test double for deterministic multi-turn testing."""

    def __init__(self) -> None:
        self.store: dict[str, list[str]] = {}

    def rpush(self, key: str, *values: str) -> int:
        self.store.setdefault(key, []).extend(values)
        return len(self.store[key])

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        lst = self.store.get(key, [])
        if not lst:
            return []
        if stop == -1:
            return list(lst[start:])
        return list(lst[start : stop + 1])

    def ltrim(self, key: str, start: int, stop: int) -> bool:
        if key not in self.store:
            return True
        lst = self.store[key]
        if start < 0:
            start = max(0, len(lst) + start)
        if stop < 0:
            stop = len(lst) + stop
        self.store[key] = lst[start : stop + 1]
        return True

    def expire(self, key: str, ttl: int) -> bool:
        return True

    def delete(self, *names: str) -> int:
        count = 0
        for n in names:
            if n in self.store:
                del self.store[n]
                count += 1
        return count

    def exists(self, key: str) -> int:
        return 1 if key in self.store else 0


class TestChatPipelineE2E(unittest.TestCase):
    """End-to-end integration tests for Phase 5 chat pipeline."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.fake_redis = FakeRedis()
        self.redis_patcher = patch(
            "app.memory.redis_memory.redis.Redis",
            return_value=self.fake_redis,
        )
        self.redis_patcher.start()

        self.db = SessionLocal()
        # Clean up any leftover test user from previous runs
        existing_user = (
            self.db.query(User)
            .filter(User.username == "e2e_regulatory_officer")
            .first()
        )
        if existing_user:
            self.db.query(ChatLog).filter(ChatLog.user_id == existing_user.id).delete()
            self.db.delete(existing_user)
            self.db.commit()

        # Create persistent test user in DB for ChatLog foreign key integrity
        self.test_user = User(
            username="e2e_regulatory_officer",
            email="e2e_officer@pharma.com",
            hashed_password="fake_hashed_secret_for_e2e",
            is_active=True,
        )
        self.db.add(self.test_user)
        self.db.commit()
        self.db.refresh(self.test_user)

        self.test_session_ids: set[str] = set()
        app.dependency_overrides = {}

    def tearDown(self) -> None:
        self.redis_patcher.stop()
        app.dependency_overrides = {}
        try:
            self.db.rollback()
            # Clean up test chat logs
            if self.test_session_ids:
                self.db.query(ChatLog).filter(
                    ChatLog.session_id.in_(list(self.test_session_ids))
                ).delete(synchronize_session=False)

            # Clean up test user
            user_in_db = self.db.get(User, self.test_user.id)
            if user_in_db:
                self.db.query(ChatLog).filter(ChatLog.user_id == user_in_db.id).delete()
                self.db.delete(user_in_db)
            self.db.commit()
        finally:
            self.db.close()

    def _set_authenticated_user(self, user: User | None = None) -> None:
        active_user = user or self.test_user
        app.dependency_overrides[get_current_user] = lambda: active_user

    # 1. Multi-turn chat + Redis memory
    @patch("app.agent.graph.rag_search")
    @patch("app.agent.graph.call_llm")
    def test_e2e_full_chat_pipeline_multi_turn_memory(
        self,
        mock_call_llm: MagicMock,
        mock_rag_search: MagicMock,
    ) -> None:
        """Verify multi-turn conversation flow preserves and recalls context across turns."""
        self._set_authenticated_user()
        session_id = "e2e-session-multi-turn"
        self.test_session_ids.add(session_id)

        mock_rag_search.return_value = "[Evidence: ICH Q1A active substance stability guidance]"

        # Turn 1
        query_1 = "What are the primary stability testing criteria for active pharmaceutical substances?"
        response_1_text = "Primary stability criteria requires 25 deg C / 60% RH long term per ICH Q1A."
        mock_call_llm.return_value = response_1_text

        r1 = self.client.post(
            "/api/v1/chat",
            json={"session_id": session_id, "message": query_1},
        )
        self.assertEqual(r1.status_code, status.HTTP_200_OK)
        body_1 = r1.json()
        self.assertEqual(body_1["session_id"], session_id)
        self.assertEqual(body_1["response"], response_1_text)

        # Verify Turn 1 is persisted in Redis memory
        mem = RedisConversationMemory(client=self.fake_redis)
        messages_turn_1 = mem.get_messages(session_id)
        self.assertEqual(len(messages_turn_1), 2)
        self.assertEqual(messages_turn_1[0].role, "user")
        self.assertEqual(messages_turn_1[0].content, query_1)
        self.assertEqual(messages_turn_1[1].role, "assistant")
        self.assertEqual(messages_turn_1[1].content, response_1_text)

        # Turn 2: Follow-up question using same session_id
        query_2 = "What are the specific parameters for accelerated testing conditions?"
        response_2_text = "Accelerated stability testing conditions require 40 deg C / 75% RH for 6 months."
        mock_call_llm.return_value = response_2_text

        r2 = self.client.post(
            "/api/v1/chat",
            json={"session_id": session_id, "message": query_2},
        )
        self.assertEqual(r2.status_code, status.HTTP_200_OK)
        body_2 = r2.json()
        self.assertEqual(body_2["session_id"], session_id)
        self.assertEqual(body_2["response"], response_2_text)

        # Verify Turn 2's LLM call received Turn 1's history in context
        self.assertEqual(mock_call_llm.call_count, 2)
        turn_2_call_args = mock_call_llm.call_args[0][0]  # messages passed to call_llm
        turn_2_contents = [m["content"] for m in turn_2_call_args]

        self.assertIn(query_1, turn_2_contents)
        self.assertIn(response_1_text, turn_2_contents)

        # Verify Redis memory now stores all 4 messages in chronological order
        messages_after_turn_2 = mem.get_messages(session_id)
        self.assertEqual(len(messages_after_turn_2), 4)
        self.assertEqual(messages_after_turn_2[0].role, "user")
        self.assertEqual(messages_after_turn_2[0].content, query_1)
        self.assertEqual(messages_after_turn_2[1].role, "assistant")
        self.assertEqual(messages_after_turn_2[1].content, response_1_text)
        self.assertEqual(messages_after_turn_2[2].role, "user")
        self.assertEqual(messages_after_turn_2[2].content, query_2)
        self.assertEqual(messages_after_turn_2[3].role, "assistant")
        self.assertEqual(messages_after_turn_2[3].content, response_2_text)

    # 2. RAG evidence flow
    @patch("app.agent.graph.call_llm")
    @patch("app.agent.graph.rag_search")
    def test_e2e_rag_search_evidence_flow(
        self,
        mock_rag_search: MagicMock,
        mock_call_llm: MagicMock,
    ) -> None:
        """Verify regulatory query executes rag_search, passes evidence, and enforces guardrail prompt."""
        self._set_authenticated_user()
        session_id = "e2e-session-rag-evidence"
        self.test_session_ids.add(session_id)

        # Query routes to rag_search
        query = "What are the regulatory requirements for stability testing of active pharmaceutical substances?"
        deterministic_evidence = (
            "[Document 101, Chunk 1] ICH Q1A(R2) Section 2.2: Stability Testing of Active Substances.\n"
            "Studies should evaluate the stability of the substance under long term (25C +/- 2C / 60% RH +/- 5% RH) "
            "and accelerated (40C +/- 2C / 75% RH +/- 5% RH) conditions."
        )
        mock_rag_search.return_value = deterministic_evidence

        synthesized_llm_answer = (
            "Based on ICH Q1A(R2) Section 2.2 (Document 101, Chunk 1), active substance stability "
            "requires long term evaluation at 25C/60% RH and accelerated testing at 40C/75% RH."
        )
        mock_call_llm.return_value = synthesized_llm_answer

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": session_id, "message": query},
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["session_id"], session_id)
        self.assertEqual(data["response"], synthesized_llm_answer)

        # Verify rag_search was executed with the user query
        mock_rag_search.assert_called_once_with(query)

        # Verify LLM was invoked with retrieved evidence and strict regulatory guardrail
        mock_call_llm.assert_called_once()
        llm_messages = mock_call_llm.call_args[0][0]

        # 1. System prompt contains strict regulatory guardrail
        system_msg = next(m["content"] for m in llm_messages if m["role"] == "system")
        self.assertIn("STRICT GUARDRAIL", system_msg)
        self.assertIn("You MUST NOT make definitive regulatory approvals", system_msg)
        self.assertIn("Final regulatory determinations remain the sole responsibility", system_msg)

        # 2. User prompt contains the supplied RAG evidence
        user_msg = next(
            m["content"] for m in llm_messages if m["role"] == "user" and "Retrieved Evidence:" in m["content"]
        )
        self.assertIn("Retrieved Evidence:", user_msg)
        self.assertIn(deterministic_evidence, user_msg)
        self.assertIn(query, user_msg)

    # 3. Drug lookup flow
    @patch("app.agent.graph.drug_lookup")
    def test_e2e_drug_lookup_flow(
        self,
        mock_drug_lookup: MagicMock,
    ) -> None:
        """Verify pharmaceutical query triggers drug_lookup and formats response with mandatory disclaimer."""
        self._set_authenticated_user()
        session_id = "e2e-session-drug-lookup"
        self.test_session_ids.add(session_id)

        query = "Look up the official drug label information for Amoxicillin"
        mock_openfda_record = {
            "status": "found",
            "source": "openFDA",
            "drug_name": "amoxicillin",
            "brand_name": "AMOXIL",
            "generic_name": "amoxicillin",
            "active_ingredients": ["AMOXICILLIN"],
            "manufacturer": "GlaxoSmithKline LLC",
            "product_type": "HUMAN PRESCRIPTION DRUG",
            "indications_and_usage": "Amoxil is indicated for bacterial infections of susceptible strains.",
            "warnings": "Serious hypersensitivity reactions have been reported.",
            "dosage_and_administration": "Administer every 8 or 12 hours as prescribed.",
            "disclaimer": (
                "This information is retrieved from official openFDA regulatory drug labeling "
                "records for informational purposes only. It does not constitute medical advice, "
                "prescription, or treatment recommendations."
            ),
        }
        mock_drug_lookup.return_value = mock_openfda_record

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": session_id, "message": query},
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["session_id"], session_id)

        # Verify drug lookup was invoked with extracted drug name
        mock_drug_lookup.assert_called_once()
        called_drug_name = mock_drug_lookup.call_args[0][0].lower()
        self.assertIn("amoxicillin", called_drug_name)

        # Verify structured drug information appears in response
        resp_text = data["response"]
        self.assertIn("Official openFDA Regulatory Drug Labeling Record: AMOXIL", resp_text)
        self.assertIn("Brand Name: AMOXIL", resp_text)
        self.assertIn("Generic Name: amoxicillin", resp_text)
        self.assertIn("Active Ingredients: AMOXICILLIN", resp_text)
        self.assertIn("Manufacturer: GlaxoSmithKline LLC", resp_text)

        # Verify mandatory medical/regulatory disclaimer is strictly present
        self.assertIn("Medical Disclaimer:", resp_text)
        self.assertIn("does not constitute medical advice", resp_text)

    # 4. ChatLog audit persistence
    @patch("app.agent.graph.rag_search")
    @patch("app.agent.graph.call_llm")
    def test_e2e_chat_log_audit_persistence(
        self,
        mock_call_llm: MagicMock,
        mock_rag_search: MagicMock,
    ) -> None:
        """Verify chat interaction is committed to PostgreSQL chat_logs table."""
        self._set_authenticated_user()
        session_id = "e2e-session-audit-persistence"
        self.test_session_ids.add(session_id)

        mock_rag_search.return_value = "[Evidence: EU MDR Annex VIII Classification Rules]"
        query = "What is the classification rule for implantable medical devices?"
        assistant_reply = "Implantable medical devices are generally categorized as Class IIb or Class III."
        mock_call_llm.return_value = assistant_reply

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": session_id, "message": query},
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Query database directly to confirm audit record creation
        audit_entry = (
            self.db.query(ChatLog)
            .filter(ChatLog.session_id == session_id)
            .first()
        )
        self.assertIsNotNone(audit_entry, "ChatLog audit record was not persisted to database.")
        self.assertEqual(audit_entry.user_id, self.test_user.id)
        self.assertEqual(audit_entry.session_id, session_id)
        self.assertEqual(audit_entry.user_query, query)
        self.assertEqual(audit_entry.assistant_response, assistant_reply)
        self.assertIsNotNone(audit_entry.created_at)

    # 5. Session isolation
    @patch("app.agent.graph.rag_search")
    @patch("app.agent.graph.call_llm")
    def test_e2e_session_isolation(
        self,
        mock_call_llm: MagicMock,
        mock_rag_search: MagicMock,
    ) -> None:
        """Verify distinct session IDs maintain isolated conversation buffers in memory."""
        self._set_authenticated_user()
        session_a = "e2e-session-alpha-isolation"
        session_b = "e2e-session-beta-isolation"
        self.test_session_ids.update([session_a, session_b])

        mock_rag_search.return_value = "[Evidence: Clinical Dossier Protocol]"

        query_a = "Confidential query regarding Project Alpha oncology dossier."
        reply_a = "Project Alpha dossier guidelines noted."
        query_b = "Independent query regarding Project Beta vaccine clinical protocol."
        reply_b = "Project Beta protocol guidelines noted."

        # Request on Session A
        mock_call_llm.return_value = reply_a
        r_a = self.client.post(
            "/api/v1/chat",
            json={"session_id": session_a, "message": query_a},
        )
        self.assertEqual(r_a.status_code, status.HTTP_200_OK)

        # Request on Session B
        mock_call_llm.return_value = reply_b
        r_b = self.client.post(
            "/api/v1/chat",
            json={"session_id": session_b, "message": query_b},
        )
        self.assertEqual(r_b.status_code, status.HTTP_200_OK)

        # Verify isolated Redis memory buffers
        mem = RedisConversationMemory(client=self.fake_redis)
        messages_a = mem.get_messages(session_a)
        messages_b = mem.get_messages(session_b)

        self.assertEqual(len(messages_a), 2)
        self.assertEqual(len(messages_b), 2)

        # Session A contains only Alpha content
        self.assertTrue(all("Alpha" in m.content for m in messages_a))
        self.assertFalse(any("Beta" in m.content for m in messages_a))

        # Session B contains only Beta content
        self.assertTrue(all("Beta" in m.content for m in messages_b))
        self.assertFalse(any("Alpha" in m.content for m in messages_b))

        # Follow-up on Session A should only contain Alpha in context
        query_a_followup = "What is the timeline for Alpha?"
        reply_a_followup = "Timeline for Alpha is Q4."
        mock_call_llm.return_value = reply_a_followup

        r_a2 = self.client.post(
            "/api/v1/chat",
            json={"session_id": session_a, "message": query_a_followup},
        )
        self.assertEqual(r_a2.status_code, status.HTTP_200_OK)

        latest_call_messages = mock_call_llm.call_args[0][0]
        latest_contents = [m["content"] for m in latest_call_messages]

        self.assertIn(query_a, latest_contents)
        self.assertIn(reply_a, latest_contents)
        self.assertFalse(any("Beta" in c for c in latest_contents), "Session B content leaked into Session A!")

    # 6. Unauthenticated rejection
    @patch("app.api.chat.run_agent")
    def test_e2e_unauthenticated_rejection(
        self,
        mock_run_agent: MagicMock,
    ) -> None:
        """Verify unauthenticated request is rejected with 401 without invoking agent."""
        # Ensure no user override is active
        app.dependency_overrides = {}

        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "e2e-session-unauth", "message": "Unauthorized attempt"},
        )
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertIn("detail", response.json())

        # Agent execution must not be invoked
        mock_run_agent.assert_not_called()

        # No chat log or memory should be created
        mem = RedisConversationMemory(client=self.fake_redis)
        self.assertEqual(len(mem.get_messages("e2e-session-unauth")), 0)


if __name__ == "__main__":
    unittest.main()
