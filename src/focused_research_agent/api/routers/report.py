"""
Report generation API endpoint for the Focused Research Agent.

This module exposes the HTTP endpoint for deep research report
generation. It receives a validated report request, obtains a
database session and executes the report use case through
dependency injection, and returns the structured report response.

Architecturally, this module belongs to the transport layer. It
stays thin — no business logic, no database queries, no graph
calls. It delegates everything to the application layer through
execute_report.
"""

from typing import Annotated, Callable
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from focused_research_agent.api.dependencies import get_report_use_case
from focused_research_agent.api.schemas.report.report import (
    ReportRequest,
    ReportResponse,
)
from focused_research_agent.database.database import get_db

report_router = APIRouter(tags=["report"])


@report_router.post(
    "/report", status_code=status.HTTP_200_OK, response_model=ReportResponse
)
def report(
    request: ReportRequest,
    db: Annotated[Session, Depends(get_db)],
    run_report_use_case: Annotated[Callable, Depends(get_report_use_case)],
) -> dict:
    """
    Handle a report generation request through the API.

    Accepts a validated report request containing a question,
    executes the deep research report use case, and returns the
    structured result with a full markdown report in the answer field.

    Args:
        request: Validated report request payload.
        db: Injected SQLAlchemy database session.
        run_report_use_case: Injected report use case callable.

    Returns:
        dict: Structured report response returned by the application
            layer. The answer field contains structured markdown with
            Introduction, Key Findings, Analysis, and Conclusion.
    """

    return run_report_use_case(
        question=request.question,
        db=db,
    )
