# Phase 8 Demo Runbook & End-to-End Validation Guide

This runbook provides step-by-step instructions for validating and demonstrating the **Regulatory Affairs Assistant** across all three core demonstration scenarios (RAG question, Tool-use openFDA lookup, and Redis conversation memory), as well as executing the complete API verification flow using Postman.

---

## A. Prerequisites

Before running the application demonstration, ensure the following host environment prerequisites are satisfied:

1. **Python 3.12 Virtual Environment:**
   Ensure the project virtual environment is active:
   ```powershell
   Set-Location D:\rf
   .\.venv\Scripts\Activate.ps1
   ```

2. **PostgreSQL 16 with pgvector:**
   PostgreSQL 16 must be running on port `5432` with the `vector` extension enabled and the schema initialized via `docker/init.sql`:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Redis:**
   A Redis server must be running and accessible on port `6379`.
   > **Note on Memory Demonstration:** Redis must be running for Demo 3 (Session Memory Recall) and session clearing to function.

4. **OpenAI API Key (`OPENAI_API_KEY`):**
   A valid OpenAI API key is required in `.env` for generating dense 1536-dimensional vector embeddings (`text-embedding-3-small`) and synthesizing LLM responses (`gpt-4o`).

5. **Environment Configuration (`.env`):**
   Ensure `.env` exists in the project root with valid configurations:
   ```env
   JWT_SECRET_KEY=your_secure_random_jwt_secret_key_here
   OPENAI_API_KEY=your_real_openai_api_key_here
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=regulatory_affairs_db
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_postgres_password
   REDIS_HOST=localhost
   REDIS_PORT=6379
   ```

6. **Environment Limitations Notice:**
   > **Docker Runtime Availability:** Docker Desktop / Docker Engine is not currently installed on this host machine. Therefore, Docker runtime commands (`docker compose up`) cannot be demonstrated locally. All demonstration steps below utilize the native local development processes.

---

## B. Start Backend Service

Start the FastAPI application using Uvicorn:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **API Base URL:** `http://127.0.0.1:8000`
- **Swagger / OpenAPI Documentation:** `http://127.0.0.1:8000/docs`
- **Operational Health Probe:** `http://127.0.0.1:8000/health`

---

## C. Start Frontend Service

In a separate terminal, launch the Streamlit frontend:

```powershell
streamlit run frontend/app.py
```

- **Frontend UI URL:** `http://localhost:8501`
- The Streamlit interface automatically connects to the FastAPI backend via `API_BASE_URL=http://127.0.0.1:8000`.

---

## D. Demo Scenario 1 — RAG Question (Regulatory Document Retrieval)

This scenario demonstrates document ingestion, pgvector similarity search, and grounded LLM synthesis with source citations.

### Steps:
1. **Ingest a Regulatory Document:**
   Upload a real regulatory guidance document (e.g., an FDA or ICH stability guidance PDF, DOCX, or TXT) via the API:
   ```http
   POST /documents/upload
   Authorization: Bearer <access_token>
   Content-Type: multipart/form-data
   (file = <path_to_guidance_file>)
   ```
2. **Wait for Processing:**
   The backend extracts text, breaks the document into 1000-character chunks (with 200-character overlap), generates embeddings, and indexes them in PostgreSQL `document_chunks`.
3. **Submit Query in Streamlit UI or Chat API:**
   Ask a factual question specific to the uploaded document, for example:
   > *"What are the long-term and accelerated storage conditions for stability testing under the guidance?"*
4. **Verify Behavior:**
   - The intent router in `app/agent/graph.py` routes the query to `rag_search`.
   - `rag_search` retrieves top matching chunks from pgvector using cosine similarity.
   - The response synthesizes answers grounded strictly in the retrieved evidence chunks.
   - Source citations appear in the text (e.g., `[Source 1: document_name.pdf (Chunk 0, Similarity: 0.88)]`).
   - In the Streamlit UI, expand the collapsible **"📚 Sources & Regulatory References"** drawer to verify cited sections.

---

## E. Demo Scenario 2 — Tool-Use Question (openFDA Public Drug Information)

This scenario demonstrates dynamic agent tool routing to the public openFDA drug labeling database without querying internal RAG documents.

### Steps:
1. **Submit Query in Streamlit UI or Chat API:**
   Submit an inquiry targeting official drug labeling:
   > *"Look up the regulatory label information for Lipitor."*
2. **Verify Behavior:**
   - The intent router detects drug-specific keywords and routes the query to `drug_lookup`.
   - The agent queries `https://api.fda.gov/drug/label.json` for brand name `"Lipitor"`.
   - The response formats structured drug facts:
     - **Brand Name:** Lipitor
     - **Generic Name:** Atorvastatin Calcium
     - **Active Ingredient:** Atorvastatin Calcium
     - **Manufacturer / Sponsor:** Parke-Davis / Viatris
     - **Indications & Usage:** Cardiovascular risk reduction, hyperlipidemia
   - The response includes the required medical and regulatory disclaimer:
     > *"Medical Disclaimer: openFDA records provide public reference data and do not substitute for formal regulatory or medical guidance."*
   - In the Streamlit UI, the source expander references `Official openFDA Regulatory Drug Labeling Database (api.fda.gov)`.

---

## F. Demo Scenario 3 — Redis Conversation Memory (Multi-Turn Session Recall)

This scenario demonstrates conversation state persistence across turns within the same session ID.

### Steps:
1. **Ensure Redis is Active:**
   Verify Redis is reachable on `localhost:6379`.
2. **Turn 1 — Establish Context:**
   Using the same session ID (e.g., `demo-session-001`), submit:
   > *"We are evaluating stability conditions for Formulation ABC-101."*
   - The agent acknowledges the formulation and stores both the user query and assistant response in Redis under `chat:memory:demo-session-001`.
3. **Turn 2 — Recall Context:**
   Using the exact same session ID (`demo-session-001`), submit:
   > *"What was the formulation code I just mentioned?"*
4. **Verify Expected Memory Behavior:**
   - `load_memory_node` in `app/agent/graph.py` loads preceding conversation messages from Redis.
   - The agent responds:
     > *"The formulation code you mentioned is Formulation ABC-101."*
   - Context is recalled accurately across independent HTTP requests.

---

## G. Complete Postman Flow

The repository includes a ready-to-import Postman collection: `postman_collection.json`.

### Recommended Execution Sequence:

1. **1. Operational Health Check (`GET /health`):**
   - Confirm backend, PostgreSQL, and Redis connectivity (`status: healthy`, HTTP 200).
2. **2. Register User (`POST /auth/register`):**
   - Register account `regulatory_specialist` (HTTP 201).
3. **3. Login & Obtain JWT Token (`POST /auth/login`):**
   - Authenticate with username and password (HTTP 200).
   - The embedded Postman test script automatically saves `access_token` to collection variables.
4. **4. Get Current User Profile (`GET /auth/me`):**
   - Verify Bearer token authorization and user details (HTTP 200).
5. **5. Upload Regulatory Document (`POST /documents/upload`):**
   - Select a local `.pdf`, `.docx`, or `.txt` file under Body form-data and upload (HTTP 201).
6. **6. Chat Query - RAG Question (`POST /api/v1/chat`):**
   - Submit a guidance question grounded in the uploaded document (HTTP 200).
7. **7. Chat Query - Tool Use (`POST /api/v1/chat`):**
   - Submit `"Look up the regulatory label information for Lipitor."` (HTTP 200).
8. **8. Chat Query - Memory Recall Turn 1 (`POST /api/v1/chat`):**
   - Send `"We are evaluating stability conditions for Formulation ABC-101."` (HTTP 200).
9. **9. Chat Query - Memory Recall Turn 2 (`POST /api/v1/chat`):**
   - Send `"What was the formulation code I just mentioned?"` using the same `session_id` (HTTP 200).
10. **10. Clear Chat Session Memory (`DELETE /api/v1/chat/{{session_id}}`):**
    - Clear Redis session memory for `demo-session-001` (HTTP 200).
    - Note that compliance audit records in PostgreSQL `chat_logs` remain preserved.

---

## H. Troubleshooting Guide

| Issue / Error | Root Cause | Resolution |
|---|---|---|
| **Backend Unavailable / Connection Refused on :8000** | FastAPI backend is not running | Start backend with `uvicorn app.main:app --host 127.0.0.1 --port 8000`. |
| **PostgreSQL Disconnected (`503 Service Unavailable` on `/health`)** | PostgreSQL service stopped or credentials incorrect in `.env` | Ensure PostgreSQL service is active on port `5432`. Verify `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` in `.env`. |
| **Redis Disconnected (`503 Service Unavailable` on `/health` or Chat)** | Redis server is not listening on port `6379` | Start local Redis server or container. Check `REDIS_HOST` and `REDIS_PORT` in `.env`. |
| **`LLM configuration error: OpenAI API key is not configured` (503)** | `OPENAI_API_KEY` missing or set to placeholder | Set a valid OpenAI API key in `.env`. |
| **`Could not validate credentials` (401 Unauthorized)** | Missing, invalid, or expired JWT access token | Re-run Request `3. Login & Obtain JWT Token` to refresh `access_token`. |
| **Document Upload Failure (`400 Bad Request`)** | File missing or unsupported extension | Ensure a valid `.pdf`, `.docx`, or `.txt` file is selected in Postman form-data or API call. |
| **Docker Unavailable Error** | Docker Engine is not installed on host | Run services natively via local virtual environment (`uvicorn` and `streamlit`) as detailed in Sections B and C. |
