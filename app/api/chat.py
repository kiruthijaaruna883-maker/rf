"""Chat API endpoints for Regulatory Affairs Assistant."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.graph import (
    AgentExecutionError,
    AgentError,
    LLMAPIError,
    LLMConfigError,
    LLMError,
    run_agent,
)
from app.api.auth import get_current_user
from app.database.models import ChatLog, User
from app.database.session import get_db
from app.memory.redis_memory import RedisConversationMemory, RedisMemoryError
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit query to Regulatory Affairs agent",
    description=(
        "Invokes the LangGraph regulatory agent with Redis conversation memory "
        "and direct tools (RAG search, drug lookup, user lookup, web search). "
        "Requires authenticated user."
    ),
)
def chat_endpoint(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ChatResponse:
    """Execute regulatory assistant query and return synthesized response."""
    clean_session_id = request.session_id
    clean_message = request.message or ""

    try:
        agent_response = run_agent(
            session_id=clean_session_id,
            query=clean_message,
            user_id=str(current_user.id),
        )
    except ValueError as exc:
        logger.warning("Invalid input for chat session %s: %s", clean_session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except LLMConfigError as exc:
        logger.error("LLM configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is not properly configured.",
        ) from exc
    except LLMAPIError as exc:
        logger.error("LLM provider API error: %s (status %s)", exc, exc.status_code)
        if exc.status_code == 429:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="LLM provider rate limit exceeded. Please try again later.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Error communicating with LLM provider.",
        ) from exc
    except LLMError as exc:
        logger.error("LLM general error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Language model generation failed.",
        ) from exc
    except RedisMemoryError as exc:
        logger.error("Redis memory failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation memory service is temporarily unavailable.",
        ) from exc
    except AgentExecutionError as exc:
        logger.error("Agent execution error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent tool execution encountered an error.",
        ) from exc
    except AgentError as exc:
        logger.error("Agent error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the chat request.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error in chat endpoint: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred processing your request.",
        ) from exc

    # Persist interaction to ChatLog audit table
    try:
        chat_log = ChatLog(
            user_id=current_user.id,
            session_id=clean_session_id,
            user_query=clean_message,
            assistant_response=agent_response,
        )
        db.add(chat_log)
        db.commit()
    except Exception as db_exc:
        logger.error("Failed to persist ChatLog audit entry: %s", db_exc)
        db.rollback()

    return ChatResponse(
        session_id=clean_session_id,
        response=agent_response,
    )

@router.delete(
    "/chat/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear conversation memory for a session",
    description=(
        "Clears active Redis conversation memory for the authenticated user session. "
        "Does not delete immutable compliance audit records."
    ),
)
def clear_chat_session(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, str]:
    """Clear Redis conversation memory for the specified session."""
    clean_session_id = session_id.strip() if isinstance(session_id, str) else ""
    if not clean_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id cannot be empty or whitespace-only.",
        )

    # Check session ownership in chat_logs if existing logs exist for this session
    existing_logs = db.query(ChatLog).filter(ChatLog.session_id == clean_session_id).all()
    if existing_logs:
        if any(log.user_id != current_user.id for log in existing_logs):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to clear this chat session.",
            )

    try:
        memory = RedisConversationMemory()
        memory.clear(clean_session_id)
    except RedisMemoryError as exc:
        logger.error("Redis memory failure clearing session %s: %s", clean_session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation memory service is temporarily unavailable.",
        ) from exc

    return {
        "session_id": clean_session_id,
        "message": "Conversation history cleared successfully.",
    }
