"""Direct agent tools for Regulatory Affairs Assistant.

Provides four core tools specified in ARCHITECTURE.md:
1. rag_search(query: str) -> str: Internal regulatory document retrieval.
2. user_lookup(user_id: str) -> dict: Internal user permissions and metadata lookup.
3. drug_lookup(drug_name: str) -> dict: Factual public drug information from openFDA.
4. web_search(query: str) -> str: External regulatory agency portal search (deferred).
"""

from typing import Any, Dict, List, Optional
import httpx
from sqlalchemy.orm import Session

from app.database.models import User
from app.database.session import SessionLocal
from app.services.retrieval import retrieve_chunks

OPENFDA_DRUG_LABEL_URL: str = "https://api.fda.gov/drug/label.json"
DEFAULT_OPENFDA_TIMEOUT: float = 10.0


def rag_search(
    query: str,
    db: Optional[Session] = None,
    top_k: int = 5,
    client: Optional[httpx.Client] = None,
) -> str:
    """Search internal regulatory vector index in PostgreSQL using pgvector.

    Retrieves top matching document sections, guidelines, and document identifiers.
    Evidence is grounded in retrieved document chunks.

    Args:
        query: Non-empty search query string.
        db: Optional active SQLAlchemy database session. If omitted, uses SessionLocal().
        top_k: Maximum number of relevant chunks to retrieve (default 5).
        client: Optional httpx.Client for embedding generation dependency injection.

    Returns:
        Formatted evidence string containing source identifiers, filenames,
        similarity scores, and chunk text.

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty or whitespace-only.")

    clean_query = query.strip()

    if db is not None:
        chunks = retrieve_chunks(db=db, query=clean_query, top_k=top_k, client=client)
    else:
        with SessionLocal() as session:
            chunks = retrieve_chunks(db=session, query=clean_query, top_k=top_k, client=client)

    if not chunks:
        return "No relevant internal regulatory documents found for the query."

    evidence_lines: List[str] = [
        f"Retrieved {len(chunks)} relevant regulatory document section(s):\n"
    ]
    for i, chunk in enumerate(chunks, start=1):
        source_name = chunk.filename or f"Document-{chunk.document_id}"
        section_meta = ""
        if chunk.metadata_json and isinstance(chunk.metadata_json, dict):
            parts = [f"{k}: {v}" for k, v in chunk.metadata_json.items() if v]
            if parts:
                section_meta = f" | {', '.join(parts)}"

        header = (
            f"[Source {i}: {source_name} | Doc ID: {chunk.document_id} | "
            f"Chunk: {chunk.chunk_index} | Similarity: {chunk.similarity:.4f}{section_meta}]"
        )
        evidence_lines.append(f"{header}\n{chunk.content}\n")

    return "\n".join(evidence_lines).strip()


def user_lookup(user_id: str, db: Optional[Session] = None) -> dict:
    """Retrieve user metadata and permissions from the PostgreSQL user registry.

    Queries the existing User model and returns only non-sensitive metadata.
    Never exposes passwords, password hashes, or security credentials.

    Args:
        user_id: Non-empty user identifier string (must represent a valid user ID).
        db: Optional active SQLAlchemy database session. If omitted, uses SessionLocal().

    Returns:
        Dictionary containing user metadata or not-found status.

    Raises:
        ValueError: If user_id is empty or whitespace-only.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id cannot be empty or whitespace-only.")

    clean_id = user_id.strip()

    try:
        numeric_id = int(clean_id)
        if numeric_id <= 0:
            return {
                "found": False,
                "user_id": clean_id,
                "message": f"User ID '{clean_id}' is invalid. User IDs must be positive integers.",
            }
    except (ValueError, TypeError):
        return {
            "found": False,
            "user_id": clean_id,
            "message": f"User ID '{clean_id}' must be an integer.",
        }

    if db is not None:
        user = db.get(User, numeric_id)
    else:
        with SessionLocal() as session:
            user = session.get(User, numeric_id)

    if user is None:
        return {
            "found": False,
            "user_id": clean_id,
            "message": f"User with ID {numeric_id} was not found in the user registry.",
        }

    return {
        "found": True,
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def drug_lookup(drug_name: str, client: Optional[httpx.Client] = None) -> dict:
    """Query official openFDA Drug Label API for factual drug information.

    Retrieves structured active ingredients, indications, warnings, and manufacturer data.
    Strictly an informational retrieval tool; does NOT provide medical advice or
    treatment recommendations.

    Args:
        drug_name: Non-empty drug brand or generic active ingredient name.
        client: Optional httpx.Client for dependency injection / mocking.

    Returns:
        Structured dictionary containing factual drug labeling data, status, and disclaimer.

    Raises:
        ValueError: If drug_name is empty or whitespace-only.
    """
    if not isinstance(drug_name, str) or not drug_name.strip():
        raise ValueError("drug_name cannot be empty or whitespace-only.")

    clean_name = drug_name.strip()
    search_query = f'openfda.brand_name:"{clean_name}" openfda.generic_name:"{clean_name}"'
    params = {"search": search_query, "limit": 1}

    disclaimer = (
        "This information is retrieved from official openFDA regulatory drug labeling "
        "records for informational purposes only. It does not constitute medical advice, "
        "prescription, or treatment recommendations."
    )

    try:
        if client is not None:
            response = client.get(OPENFDA_DRUG_LABEL_URL, params=params, timeout=DEFAULT_OPENFDA_TIMEOUT)
        else:
            with httpx.Client(timeout=DEFAULT_OPENFDA_TIMEOUT) as http_client:
                response = http_client.get(OPENFDA_DRUG_LABEL_URL, params=params)
    except httpx.TimeoutException:
        return {
            "status": "error",
            "source": "openFDA",
            "drug_name": clean_name,
            "error": "Request timed out",
            "message": "The openFDA API request timed out.",
            "disclaimer": disclaimer,
        }
    except (httpx.ConnectError, httpx.RequestError) as exc:
        return {
            "status": "error",
            "source": "openFDA",
            "drug_name": clean_name,
            "error": exc.__class__.__name__,
            "message": "Network error communicating with the openFDA Drug Label API.",
            "disclaimer": disclaimer,
        }

    if response.status_code == 404:
        return {
            "status": "not_found",
            "source": "openFDA",
            "drug_name": clean_name,
            "message": f"No regulatory drug labeling records found for '{clean_name}' in openFDA.",
            "disclaimer": disclaimer,
        }
    elif response.status_code != 200:
        return {
            "status": "error",
            "source": "openFDA",
            "drug_name": clean_name,
            "error": f"HTTP {response.status_code}",
            "message": f"openFDA API returned HTTP status {response.status_code}.",
            "disclaimer": disclaimer,
        }

    try:
        data = response.json()
    except Exception:
        return {
            "status": "error",
            "source": "openFDA",
            "drug_name": clean_name,
            "error": "JSONDecodeError",
            "message": "Failed to parse response JSON from openFDA API.",
            "disclaimer": disclaimer,
        }

    results = data.get("results")
    if not results or not isinstance(results, list):
        return {
            "status": "not_found",
            "source": "openFDA",
            "drug_name": clean_name,
            "message": f"No regulatory drug labeling records found for '{clean_name}' in openFDA.",
            "disclaimer": disclaimer,
        }

    item = results[0]
    openfda = item.get("openfda", {})

    brand_names = openfda.get("brand_name", [])
    generic_names = openfda.get("generic_name", [])
    substances = openfda.get("substance_name", [])
    manufacturers = openfda.get("manufacturer_name", [])
    product_types = openfda.get("product_type", [])

    indications = item.get("indications_and_usage", [])
    warnings = item.get("warnings", []) or item.get("warnings_and_cautions", [])
    dosage = item.get("dosage_and_administration", [])

    return {
        "status": "found",
        "source": "openFDA",
        "drug_name": clean_name,
        "brand_name": brand_names[0] if brand_names else clean_name,
        "generic_name": generic_names[0] if generic_names else None,
        "active_ingredients": substances,
        "manufacturer": manufacturers[0] if manufacturers else None,
        "product_type": product_types[0] if product_types else None,
        "indications_and_usage": indications[0] if indications else None,
        "warnings": warnings[0] if warnings else None,
        "dosage_and_administration": dosage[0] if dosage else None,
        "disclaimer": disclaimer,
    }


def web_search(query: str) -> str:
    """Perform real-time external search against official regulatory agency portals.

    Notice: Real-time external search provider is intentionally unconfigured in this
    development phase. This capability is deferred until the dedicated regulatory
    search provider is integrated.

    Args:
        query: Non-empty search query string.

    Returns:
        Deterministic message stating that web search is currently deferred.

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty or whitespace-only.")

    return (
        "Web search capability is currently not configured. "
        "Real-time external regulatory portal search will be enabled in a future release."
    )
