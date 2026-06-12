"""
Research API endpoints for the Focused Research Agent.

This module exposes HTTP endpoints for the research use case. It receives
validated API input, forwards execution to the application layer through
FastAPI dependency wiring, and returns the resulting response as an
HTTP response.

Architecturally, this module belongs to the transport layer. Routers are
transport adapters and should stay thin. They should not contain workflow
orchestration or provider-specific logic.
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, status

from focused_research_agent.api.dependencies import get_research_use_case
from focused_research_agent.api.schemas.research import research as research_schema

research_router = APIRouter(tags=["research"])


@research_router.post(
    "/research",
    status_code=status.HTTP_200_OK,
    response_model=research_schema.ResearchResponse,
)
def research(
    search: research_schema.ResearchRequest,
    run_research_use_case: Annotated[
        Callable[[str], dict],
        Depends(get_research_use_case),
    ],
) -> dict:
    """
    Handle a research request through the API.

    This endpoint accepts a validated research request, obtains the
    application-layer research use case through dependency injection,
    executes that use case with the user's question, and returns the
    structured research result.

    Args:
        search: Validated research request payload.
        run_research_use_case: Injected callable that executes the
            research use case.

    Returns:
        dict: Structured research response returned by the application
        layer.
    """
    search_result = run_research_use_case(search.question)
    return search_result
