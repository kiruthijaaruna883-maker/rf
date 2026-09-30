"""LangGraph agent orchestration for Regulatory Affairs Assistant."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.state import AgentState
from app.agent.tools import drug_lookup, rag_search, user_lookup, web_search
from app.config import get_settings
from app.memory.redis_memory import ConversationMessage, RedisConversationMemory, RedisMemoryError

logger = logging.getLogger(__name__)

OPENAI_CHAT_COMPLETIONS_URL: str = "https://api.openai.com/v1/chat/completions"
DEFAULT_LLM_TIMEOUT: float = 60.0


class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class LLMError(AgentError):
    """Exception raised for LLM communication or parsing failures."""
    pass


class LLMConfigError(LLMError):
    """Exception raised when LLM is not properly configured."""
    pass


class LLMAPIError(LLMError):
    """Exception raised when LLM API returns an error response."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class AgentExecutionError(AgentError):
    """Exception raised during graph execution."""
    pass


def _validate_llm_api_key(api_key: Optional[str]) -> str:
    """Validate that the LLM API key is present and not a placeholder."""
    if not api_key or not api_key.strip():
        raise LLMConfigError(
            "OpenAI API key is not configured. Please set OPENAI_API_KEY in the environment or .env file."
        )
    cleaned = api_key.strip()
    if cleaned.startswith("your-") or "placeholder" in cleaned.lower():
        raise LLMConfigError(
            "OpenAI API key is set to a placeholder value. A valid API key is required for live LLM response generation."
        )
    return cleaned


def call_llm(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Optional[httpx.Client] = None,
    timeout: float = DEFAULT_LLM_TIMEOUT,
) -> str:
    """Send chat completion request to OpenAI-compatible endpoint."""
    settings = get_settings()
    key = _validate_llm_api_key(api_key or settings.OPENAI_API_KEY)
    resolved_model = model or settings.LLM_MODEL or "gpt-4o"
    if resolved_model.startswith("your-") or "placeholder" in resolved_model.lower():
        resolved_model = "gpt-4o"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": resolved_model,
        "messages": messages,
        "temperature": 0.0,
    }

    try:
        if client is not None:
            response = client.post(OPENAI_CHAT_COMPLETIONS_URL, json=payload, headers=headers, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as internal_client:
                response = internal_client.post(OPENAI_CHAT_COMPLETIONS_URL, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise LLMError("LLM API request timed out.") from exc
    except httpx.ConnectError as exc:
        raise LLMError("Failed to connect to LLM provider.") from exc
    except httpx.RequestError as exc:
        raise LLMError(f"Network error during LLM request: {exc.__class__.__name__}.") from exc

    if response.status_code == 401:
        raise LLMAPIError("Authentication failed: invalid or unauthorized OpenAI API key.", status_code=401)
    elif response.status_code == 429:
        raise LLMAPIError("Rate limit exceeded from LLM provider.", status_code=429)
    elif 400 <= response.status_code < 500:
        raise LLMAPIError(f"LLM API client error ({response.status_code}): {response.text}", status_code=response.status_code)
    elif response.status_code >= 500:
        raise LLMAPIError(f"LLM provider server error ({response.status_code}).", status_code=response.status_code)

    try:
        data = response.json()
        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise LLMError("Malformed LLM response: missing 'choices' array.")
        message_obj = choices[0].get("message", {})
        content = message_obj.get("content")
        if content is None:
            raise LLMError("Malformed LLM response: missing message 'content'.")
        return str(content).strip()
    except Exception as exc:
        if isinstance(exc, LLMError):
            raise
        raise LLMError(f"Failed to parse LLM response JSON: {exc}") from exc


def extract_drug_name(query: str) -> str:
    """Extract drug name candidate from query string."""
    lower_q = query.lower()
    prefixes = [
        "look up the regulatory label information for ",
        "look up regulatory label information for ",
        "regulatory label information for ",
        "what does openfda list for ",
        "openfda list for ",
        "drug lookup for ",
        "drug lookup ",
        "information for ",
        "information on ",
        "label for ",
        "list for ",
        "look up ",
        "lookup ",
        "about ",
        "for ",
        "drug ",
    ]
    for prefix in prefixes:
        idx = lower_q.find(prefix)
        if idx != -1:
            candidate = query[idx + len(prefix):].strip(" ?.!,;:'\"")
            for suffix in [" in openfda", " on openfda", " from openfda"]:
                if candidate.lower().endswith(suffix):
                    candidate = candidate[:-len(suffix)].strip()
            if candidate:
                return candidate
    return query.strip(" ?.!,;:'\"")


def route_intent(query: str, user_id: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """Determine tool to invoke based on user query and context."""
    q_lower = query.lower()

    # 1. External regulatory web search check
    web_keywords = [
        "web search",
        "external search",
        "external web",
        "recent announcements",
        "agency announcements",
        "external regulatory",
    ]
    if any(k in q_lower for k in web_keywords):
        return "web_search", query.strip()

    # 2. User information lookup check
    user_keywords = [
        "user lookup",
        "user profile",
        "user registry",
        "user account",
        "user permissions",
        "lookup user",
    ]
    if any(k in q_lower for k in user_keywords) or (
        "user" in q_lower and any(w in q_lower for w in ["profile", "registry", "permission", "account", "info", "metadata"])
    ):
        target_uid = user_id
        if not target_uid:
            m = re.search(r"user\s+(?:id\s+)?(?:#\s*)?(\d+)", q_lower)
            if m:
                target_uid = m.group(1)
        if target_uid:
            return "user_lookup", str(target_uid).strip()

    # 3. Drug information lookup check
    drug_keywords = [
        "openfda",
        "drug lookup",
        "drug label",
        "label information",
        "drug information",
        "medication",
        "active ingredient",
        "drug",
    ]
    drug_patterns = [
        r"openfda\s+list\s+for\s+([a-zA-Z0-9\-_ ]+)",
        r"(?:regulatory\s+)?label\s+information\s+for\s+([a-zA-Z0-9\-_ ]+)",
        r"information\s+(?:for|on)\s+([a-zA-Z0-9\-_ ]+)",
        r"drug\s+(?:lookup\s+for\s+|info\s+for\s+|information\s+for\s+)?([a-zA-Z0-9\-_ ]+)",
        r"look\s+up\s+(?:the\s+)?(?:regulatory\s+)?(?:drug\s+)?label\s+(?:information\s+)?for\s+([a-zA-Z0-9\-_ ]+)",
    ]
    if any(k in q_lower for k in drug_keywords) or any(re.search(p, q_lower) for p in drug_patterns):
        drug_name = extract_drug_name(query)
        return "drug_lookup", drug_name

    # 4. Direct conversational check (no tool)
    conversational_greetings = (
        "hello",
        "hi",
        "hey",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "help",
        "who are you",
        "what can you do",
    )
    cleaned_q = q_lower.strip(" ?.!,;:'\"")
    if any(cleaned_q == g or cleaned_q.startswith(g + " ") for g in conversational_greetings):
        return None, None

    # 5. Regulatory document query / RAG search (default for domain questions)
    return "rag_search", query.strip()


def format_drug_response(data: dict) -> str:
    """Format structured openFDA response with medical disclaimer."""
    status = data.get("status")
    disclaimer = data.get("disclaimer", "")
    drug_name = data.get("drug_name", "Unknown drug")

    if status == "not_found":
        msg = data.get("message", f"No regulatory drug labeling records found for '{drug_name}' in openFDA.")
        return f"{msg}\n\nMedical Disclaimer: {disclaimer}"
    elif status == "error":
        msg = data.get("message", "Error querying drug registry.")
        return f"Error retrieving drug records for '{drug_name}': {msg}\n\nMedical Disclaimer: {disclaimer}"

    lines = [
        f"Official openFDA Regulatory Drug Labeling Record: {data.get('brand_name') or drug_name}",
        f"* Brand Name: {data.get('brand_name') or 'N/A'}",
        f"* Generic Name: {data.get('generic_name') or 'N/A'}",
        f"* Active Ingredients: {', '.join(data.get('active_ingredients') or []) or 'N/A'}",
        f"* Manufacturer: {data.get('manufacturer') or 'N/A'}",
        f"* Product Type: {data.get('product_type') or 'N/A'}",
    ]
    if data.get("indications_and_usage"):
        lines.append(f"\nIndications & Usage:\n{data['indications_and_usage']}")
    if data.get("warnings"):
        lines.append(f"\nWarnings & Precautions:\n{data['warnings']}")
    if data.get("dosage_and_administration"):
        lines.append(f"\nDosage & Administration:\n{data['dosage_and_administration']}")
    if disclaimer:
        lines.append(f"\nMedical Disclaimer:\n{disclaimer}")

    return "\n".join(lines)


def format_user_response(data: dict) -> str:
    """Format non-sensitive user metadata."""
    if not data.get("found"):
        return data.get("message", "User was not found in the user registry.")

    lines = [
        f"User Registry Profile (User ID: {data.get('user_id')}):",
        f"* Username: {data.get('username')}",
        f"* Email: {data.get('email')}",
        f"* Active: {data.get('is_active')}",
        f"* Created: {data.get('created_at') or 'N/A'}",
    ]
    return "\n".join(lines)


def create_agent_graph(
    memory: Optional[RedisConversationMemory] = None,
    http_client: Optional[httpx.Client] = None,
) -> CompiledStateGraph:
    """Build and compile the LangGraph agent StateGraph.

    Args:
        memory: Optional RedisConversationMemory instance (uses default if omitted).
        http_client: Optional httpx.Client for LLM dependency injection.

    Returns:
        Compiled StateGraph ready for execution.
    """
    memory_client = memory if memory is not None else RedisConversationMemory()

    def load_memory_node(state: AgentState) -> dict[str, Any]:
        session_id = state["session_id"]
        try:
            msgs = memory_client.get_messages(session_id)
            return {"messages": msgs}
        except RedisMemoryError as exc:
            logger.error("Redis memory failure loading session %s: %s", session_id, exc)
            raise

    def agent_router_node(state: AgentState) -> dict[str, Any]:
        tool_name, tool_input = route_intent(state["query"], user_id=state.get("user_id"))
        return {"tool_name": tool_name, "tool_input": tool_input}

    def execute_tool_node(state: AgentState) -> dict[str, Any]:
        tool_name = state.get("tool_name")
        tool_input = state.get("tool_input")
        try:
            if tool_name == "rag_search":
                output = rag_search(tool_input or state["query"])
            elif tool_name == "drug_lookup":
                output = drug_lookup(tool_input or state["query"])
            elif tool_name == "user_lookup":
                output = user_lookup(tool_input or state.get("user_id") or "")
            elif tool_name == "web_search":
                output = web_search(tool_input or state["query"])
            else:
                output = None
            return {"tool_output": output}
        except Exception as exc:
            logger.error("Tool execution failed for %s: %s", tool_name, exc)
            raise AgentExecutionError(f"Error executing tool '{tool_name}': {exc}") from exc

    def generate_response_node(state: AgentState) -> dict[str, Any]:
        query = state["query"]
        session_id = state["session_id"]
        tool_name = state.get("tool_name")
        tool_output = state.get("tool_output")
        messages = state.get("messages", [])

        if tool_name == "web_search":
            response_text = str(tool_output)
        elif tool_name == "user_lookup":
            if isinstance(tool_output, dict):
                response_text = format_user_response(tool_output)
            else:
                response_text = str(tool_output)
        elif tool_name == "drug_lookup":
            if isinstance(tool_output, dict):
                response_text = format_drug_response(tool_output)
            else:
                response_text = str(tool_output)
        elif tool_name == "rag_search":
            evidence = str(tool_output) if tool_output is not None else ""
            if "No relevant regulatory guidance sections were found" in evidence:
                response_text = (
                    f"{evidence}\n\n"
                    "Notice: No grounded internal guidance matched your query. "
                    "Please verify that the required regulatory documents are ingested."
                )
            else:
                system_prompt = (
                    "You are an expert Regulatory Affairs Assistant. Synthesize an answer to the "
                    "user query based ONLY on the provided regulatory document evidence.\n"
                    "Cite specific sources (Document ID, Chunk, Source name) from the evidence.\n"
                    "STRICT GUARDRAIL: You are an informational assistant only. You MUST NOT make "
                    "definitive regulatory approvals, authorizations, or rejections. Final regulatory "
                    "determinations remain the sole responsibility of human regulatory professionals.\n"
                    "If the evidence does not contain sufficient details to answer, state that clearly."
                )
                llm_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
                for msg in messages[-5:]:
                    llm_messages.append({"role": msg.role, "content": msg.content})
                user_content = f"Retrieved Evidence:\n{evidence}\n\nUser Query: {query}"
                llm_messages.append({"role": "user", "content": user_content})

                response_text = call_llm(llm_messages, client=http_client)
        else:
            system_prompt = (
                "You are a helpful Regulatory Affairs Assistant. You assist regulatory affairs "
                "professionals by retrieving internal guidance documents, looking up official "
                "drug labeling information, and providing regulatory context. "
                "Respond politely and informatively."
            )
            llm_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages[-5:]:
                llm_messages.append({"role": msg.role, "content": msg.content})
            llm_messages.append({"role": "user", "content": query})

            response_text = call_llm(llm_messages, client=http_client)

        try:
            memory_client.add_message(session_id, "user", query)
            memory_client.add_message(session_id, "assistant", response_text)
        except RedisMemoryError as exc:
            logger.error("Redis memory failure saving message for session %s: %s", session_id, exc)
            raise

        return {"response": response_text}

    def route_tool_or_response(state: AgentState) -> str:
        if state.get("tool_name"):
            return "execute_tool"
        return "response"

    builder = StateGraph(AgentState)

    builder.add_node("load_memory", load_memory_node)
    builder.add_node("agent", agent_router_node)
    builder.add_node("execute_tool", execute_tool_node)
    builder.add_node("response", generate_response_node)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "agent")
    builder.add_conditional_edges(
        "agent",
        route_tool_or_response,
        {
            "execute_tool": "execute_tool",
            "response": "response",
        },
    )
    builder.add_edge("execute_tool", "response")
    builder.add_edge("response", END)

    return builder.compile()


# Default pre-compiled graph for static analysis and inspection
agent_graph = create_agent_graph()


def run_agent(
    session_id: str,
    query: str,
    user_id: Optional[str] = None,
    memory: Optional[RedisConversationMemory] = None,
    http_client: Optional[httpx.Client] = None,
) -> str:
    """Execute the agent graph for a user query within a chat session.

    Args:
        session_id: Non-empty chat session identifier.
        query: Non-empty user query string.
        user_id: Optional user identifier context.
        memory: Optional injected RedisConversationMemory instance.
        http_client: Optional injected httpx.Client for LLM calls.

    Returns:
        Final assistant response string.

    Raises:
        ValueError: If session_id or query is invalid/empty.
        AgentError: On graph execution, LLM, or memory failures.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    clean_session_id = session_id.strip()
    clean_query = query.strip()
    clean_user_id = user_id.strip() if isinstance(user_id, str) and user_id.strip() else None

    graph = create_agent_graph(memory=memory, http_client=http_client)

    initial_state: AgentState = {
        "session_id": clean_session_id,
        "query": clean_query,
        "user_id": clean_user_id,
        "messages": [],
        "tool_name": None,
        "tool_input": None,
        "tool_output": None,
        "response": "",
    }

    result = graph.invoke(initial_state)
    return result["response"]
