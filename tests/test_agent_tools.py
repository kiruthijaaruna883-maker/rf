"""Unit and integration tests for direct agent tools."""

from unittest.mock import MagicMock, patch
import unittest
import httpx

from app.agent.tools import drug_lookup, rag_search, user_lookup, web_search
from app.database.models import User
from app.services.retrieval import ChunkSearchResult


class TestAgentTools(unittest.TestCase):
    # -------------------------------------------------------------------------
    # rag_search tests
    # -------------------------------------------------------------------------
    def test_rag_search_empty_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            rag_search("")
        with self.assertRaises(ValueError):
            rag_search("   \n\t  ")

    @patch("app.agent.tools.retrieve_chunks")
    def test_rag_search_delegates_and_formats_evidence(self, mock_retrieve):
        mock_retrieve.return_value = [
            ChunkSearchResult(
                chunk_id=101,
                document_id=5,
                chunk_index=0,
                content="Accelerated stability testing shall be conducted at 40 degrees C.",
                distance=0.1,
                similarity=0.9,
                metadata_json={"section": "2.1"},
                filename="stability_guidance.pdf",
            ),
            ChunkSearchResult(
                chunk_id=102,
                document_id=5,
                chunk_index=1,
                content="Testing must be conducted for a minimum duration of six months.",
                distance=0.2,
                similarity=0.8,
                metadata_json={"section": "2.2"},
                filename="stability_guidance.pdf",
            ),
        ]

        result = rag_search("stability conditions")

        mock_retrieve.assert_called_once()
        self.assertIn("Retrieved 2 relevant regulatory document section(s):", result)
        self.assertIn("stability_guidance.pdf", result)
        self.assertIn("Doc ID: 5", result)
        self.assertIn("Chunk: 0", result)
        self.assertIn("Similarity: 0.9000", result)
        self.assertIn("section: 2.1", result)
        self.assertIn("40 degrees C", result)
        self.assertIn("six months", result)

    @patch("app.agent.tools.retrieve_chunks")
    def test_rag_search_no_evidence_returns_clear_message(self, mock_retrieve):
        mock_retrieve.return_value = []
        result = rag_search("non-existent regulatory topic")
        self.assertEqual(result, "No relevant internal regulatory documents found for the query.")

    # -------------------------------------------------------------------------
    # user_lookup tests
    # -------------------------------------------------------------------------
    def test_user_lookup_empty_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            user_lookup("")
        with self.assertRaises(ValueError):
            user_lookup("   ")

    def test_user_lookup_invalid_id_format(self):
        result = user_lookup("not-a-number")
        self.assertFalse(result["found"])
        self.assertIn("must be an integer", result["message"])

        result_negative = user_lookup("-5")
        self.assertFalse(result_negative["found"])
        self.assertIn("positive integers", result_negative["message"])

    def test_user_lookup_not_found(self):
        mock_db = MagicMock()
        mock_db.get.return_value = None

        result = user_lookup("999", db=mock_db)
        self.assertFalse(result["found"])
        self.assertEqual(result["user_id"], "999")
        self.assertIn("not found", result["message"].lower())

    def test_user_lookup_existing_user_metadata_and_no_credentials(self):
        mock_user = MagicMock(spec=User)
        mock_user.id = 42
        mock_user.username = "regulatory_officer"
        mock_user.email = "officer@pharma.com"
        mock_user.is_active = True
        mock_user.created_at = None
        # Ensure model has hashed_password attribute on mock to verify exclusion
        mock_user.hashed_password = "$2b$12$secret_hash_value"

        mock_db = MagicMock()
        mock_db.get.return_value = mock_user

        result = user_lookup("42", db=mock_db)

        self.assertTrue(result["found"])
        self.assertEqual(result["user_id"], 42)
        self.assertEqual(result["username"], "regulatory_officer")
        self.assertEqual(result["email"], "officer@pharma.com")
        self.assertTrue(result["is_active"])

        # Security assertion: password and credential fields must NEVER be returned
        self.assertNotIn("hashed_password", result)
        self.assertNotIn("password", result)
        self.assertNotIn("credentials", result)
        self.assertNotIn("token", result)

    # -------------------------------------------------------------------------
    # drug_lookup tests
    # -------------------------------------------------------------------------
    def test_drug_lookup_empty_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            drug_lookup("")
        with self.assertRaises(ValueError):
            drug_lookup("   \t  ")

    def test_drug_lookup_success_openfda_response(self):
        mock_json = {
            "results": [
                {
                    "openfda": {
                        "brand_name": ["Advil"],
                        "generic_name": ["IBUPROFEN"],
                        "substance_name": ["IBUPROFEN"],
                        "manufacturer_name": ["Pfizer Laboratories Div Pfizer Inc"],
                        "product_type": ["HUMAN OTC DRUG"],
                    },
                    "indications_and_usage": ["Temporarily relieves minor aches and pains."],
                    "warnings": ["Stomach bleeding warning: This product contains an NSAID."],
                    "dosage_and_administration": ["Take 1 tablet every 4 to 6 hours."],
                }
            ]
        }
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = httpx.Response(200, json=mock_json, request=httpx.Request("GET", "https://api.fda.gov/drug/label.json"))

        result = drug_lookup("advil", client=mock_client)

        self.assertEqual(result["status"], "found")
        self.assertEqual(result["source"], "openFDA")
        self.assertEqual(result["brand_name"], "Advil")
        self.assertEqual(result["generic_name"], "IBUPROFEN")
        self.assertEqual(result["active_ingredients"], ["IBUPROFEN"])
        self.assertEqual(result["manufacturer"], "Pfizer Laboratories Div Pfizer Inc")
        self.assertIn("aches and pains", result["indications_and_usage"])
        self.assertIn("disclaimer", result)

    def test_drug_lookup_not_found_response(self):
        # openFDA returns 404 for not found
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = httpx.Response(
            404,
            json={"error": {"code": "NOT_FOUND", "message": "No matches found!"}},
            request=httpx.Request("GET", "https://api.fda.gov/drug/label.json"),
        )

        result = drug_lookup("unknownfakechemical123", client=mock_client)

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["source"], "openFDA")
        self.assertIn("No regulatory drug labeling records found", result["message"])

    def test_drug_lookup_http_error(self):
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = httpx.Response(
            500,
            json={"error": "Internal Server Error"},
            request=httpx.Request("GET", "https://api.fda.gov/drug/label.json"),
        )

        result = drug_lookup("aspirin", client=mock_client)

        self.assertEqual(result["status"], "error")
        self.assertIn("500", result["error"])

    def test_drug_lookup_timeout_and_network_error(self):
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.TimeoutException("Read timed out")

        result = drug_lookup("aspirin", client=mock_client)

        self.assertEqual(result["status"], "error")
        self.assertIn("timed out", result["message"].lower())

    def test_drug_lookup_malformed_json_response(self):
        mock_client = MagicMock(spec=httpx.Client)
        # Invalid JSON payload
        mock_resp = httpx.Response(200, content=b"Invalid JSON{", request=httpx.Request("GET", "https://api.fda.gov/drug/label.json"))
        mock_client.get.return_value = mock_resp

        result = drug_lookup("aspirin", client=mock_client)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "JSONDecodeError")

    # -------------------------------------------------------------------------
    # web_search tests
    # -------------------------------------------------------------------------
    def test_web_search_empty_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            web_search("")
        with self.assertRaises(ValueError):
            web_search("   ")

    def test_web_search_returns_unconfigured_message(self):
        result = web_search("FDA accelerated approval guidelines 2026")
        self.assertIn("Web search capability is currently not configured", result)
        self.assertIn("future release", result)


if __name__ == "__main__":
    unittest.main()
