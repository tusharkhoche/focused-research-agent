"""
Health-check endpoints for the Focused Research Agent API.

This module defines lightweight endpoints used to verify that the FastAPI
service is running and reachable.

Architecturally, routers are transport adapters. They should remain thin and
handle HTTP concerns such as routes, request/response mapping, and status
codes, while avoiding business logic.
"""

from fastapi import APIRouter

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
def health() -> dict:
    """
    Return a simple health status for the API service.

    This endpoint is used to confirm that the FastAPI application is up
    and able to receive requests.

    Returns:
        dict: A minimal status payload indicating that the service is
            healthy.
    """
    return {"status": "ok"}
