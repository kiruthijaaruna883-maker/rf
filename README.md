# Regulatory Affairs Assistant

An intelligent, AI-powered decision-support system designed to assist regulatory affairs and life sciences professionals with regulatory compliance analysis, internal document retrieval, openFDA drug labeling queries, and grounded guidance synthesis.

**Current Status:** Phase 7 — Production Readiness, Containerization & Operations

---

## Architecture Overview

The system is organized into three distinct architectural tiers:

```
[ User Browser ]
       ↓ HTTP :8501
[ Streamlit Frontend (frontend/) ]
       ↓ REST API (Bearer JWT) :8000
[ FastAPI Application (app/) ]
       ↓ StateGraph Orchestration
[ LangGraph Agent (app/agent/) ]
       ├── Direct Tools (rag_search, drug_lookup, user_lookup, web_search stub)
       ├── Redis Conversation Memory (chat:memory:{session_id})
       ├── PostgreSQL pgvector (similarity search on document_chunks)
       └── PostgreSQL Compliance Audit (chat_logs)
```

1. **Presentation Layer (Streamlit):** Web user interface providing tabbed authentication (Login/Registration), active session controls, markdown message streaming, and collapsible source citation breakdowns.
2. **Application Layer (FastAPI & LangGraph):** High-performance RESTful API enforcing JWT authentication, input validation, conversation session memory clearing (`DELETE /api/v1/chat/{session_id}`), operational health checks (`GET /health`), and LangGraph StateGraph reasoning loops.
3. **Data & Cache Layer (PostgreSQL 16 & Redis):** PostgreSQL with `pgvector` extension for dense 1536-dimensional vector similarity retrieval and relational user/audit storage; Redis for bounded (50 messages) 24-hour TTL session memory.

---

## Technology Stack

| Layer | Component | Version / Technology |
|---|---|---|
| **Frontend** | Streamlit | 1.60.0 |
| **Backend API** | FastAPI / Uvicorn | 0.141.1 / 0.54.0 |
| **Agent Orchestration** | LangGraph | 1.2.12 |
| **Relational Database & Vectors** | PostgreSQL 16 + pgvector | 16.15 / 0.8.6 |
| **Conversation Memory Store** | Redis | 7+ / 8.10.1 |
| **Security & Auth** | PyJWT / bcrypt | 2.15.0 / 5.0.0 (HS256) |
| **Embeddings & LLM** | OpenAI (direct HTTP via httpx) | text-embedding-3-small / gpt-4o |
| **Configuration** | Pydantic Settings | 2.15.0 |

---

## Regulatory Safety Notice

> **Informational Guardrail**: This application is strictly an informational decision-support assistant. It provides research references and regulatory guidance analysis based on ingested documents and openFDA records. It does **not** provide medical advice, formulate treatment decisions, or autonomously approve, reject, or authorize regulatory filings. Final regulatory determinations remain the sole responsibility of qualified human regulatory professionals.

---

## Prerequisites

- **Python:** 3.12+
- **PostgreSQL:** Version 16 with `pgvector` extension enabled
- **Redis:** Version 7.0+
- **Docker Desktop:** Version 20.10+ (for containerized deployment)

---

## Local Development Setup

### 1. Clone & Virtual Environment

```powershell
Set-Location D:
f
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the template environment file:

```powershell
Copy-Item .env.example .env
```

Configure your secrets in `.env`:
- `JWT_SECRET_KEY`: Random 256-bit string
- `OPENAI_API_KEY`: Valid OpenAI API key (required for live embedding generation and LLM synthesis)
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`: PostgreSQL credentials
- `REDIS_HOST`, `REDIS_PORT`: Redis connection details

### 3. Database Initialization

Ensure PostgreSQL 16 is running with the `vector` extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Execute the database schema creation script:

```powershell
psql -U postgres -d regulatory_affairs_db -f docker/init.sql
```

### 4. Running the Application

**Terminal 1 — FastAPI Backend:**
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Streamlit Frontend:**
```powershell
streamlit run frontend/app.py
```

Access the Streamlit UI at `http://localhost:8501`.
Access Swagger API documentation at `http://localhost:8000/docs`.

---

## Operational Health Probe

The application exposes a dedicated, unauthenticated operational probe:

```http
GET /health
```

**Sample Response (HTTP 200 OK):**
```json
{
  "status": "healthy",
  "database": "healthy",
  "redis": "healthy"
}
```

If either PostgreSQL or Redis connectivity fails, the endpoint returns `503 Service Unavailable` with `"status": "unhealthy"` indicating the affected component.

---

## Docker Compose Deployment

The entire application stack (PostgreSQL + pgvector, Redis, FastAPI, Streamlit) is fully containerized and orchestrated via Docker Compose.

### Starting the Stack

```bash
docker compose up --build -d
```

### Stopping the Stack

```bash
docker compose down
```

### Services Orchestrated

1. **`db` (pgvector/pgvector:pg16):** PostgreSQL 16 with pgvector pre-installed on port 5432. Mounts `docker/init.sql` into `/docker-entrypoint-initdb.d/init.sql` for automated schema provisioning on initial volume creation.
2. **`redis` (redis:7-alpine):** Redis in-memory cache and session store on port 6379.
3. **`backend` (Dockerfile.backend):** FastAPI application listening on port 8000. Waits for `db` and `redis` to pass health checks before starting.
4. **`frontend` (Dockerfile.frontend):** Streamlit web application on port 8501. Communicates server-to-server with FastAPI at `http://backend:8000`.

### Recreating a Fresh Database Volume

PostgreSQL initialization scripts in `/docker-entrypoint-initdb.d/` execute **only** when the database volume is initialized for the first time. To reset the database volume and re-run `docker/init.sql`:

```bash
# WARNING: This deletes existing data volumes
docker compose down -v
docker compose up --build -d
```

---

## Testing & Quality Assurance

The project's official test runner is Python's standard library `unittest`:

```powershell
# Run the complete test suite (177 tests)
python -m unittest discover -s tests -q

# Validate bytecode compilation
python -m compileall -q app frontend tests

# Validate package dependency consistency
python -m pip check

# Validate Git formatting
git diff --check
```

---

## Environment Variables Reference

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_HOST` | `localhost` (Compose: `db`) | PostgreSQL server hostname |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `regulatory_affairs_db` | Application database name |
| `POSTGRES_USER` | `postgres` | Database username |
| `POSTGRES_PASSWORD` | `""` | Database password |
| `REDIS_HOST` | `localhost` (Compose: `redis`) | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | `None` | Optional Redis password |
| `JWT_SECRET_KEY` | *(Required)* | Secret key for signing HS256 tokens |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access token lifespan |
| `OPENAI_API_KEY` | *(Required for live LLM)* | OpenAI API key |
| `LLM_MODEL` | `gpt-4o` | Language model for synthesis |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Dense embedding model (1536 dim) |
| `API_BASE_URL` | `http://127.0.0.1:8000` (Compose: `http://backend:8000`) | FastAPI target for Streamlit |
