"""
HTTP client for the Focused Research Agent Streamlit UI.

This module is the only file in the UI layer that knows about httpx.
It calls the FastAPI backend and returns plain Python dicts to the caller.
It contains no Streamlit code.

Architecturally, this module is an external integration adapter for the UI
transport layer — the same role search_provider_tavily.py plays for the
search integration, but pointing at the internal FastAPI backend instead
of an external API.
"""

from typing import TypedDict
import httpx
from focused_research_agent.config.ui_config import get_ui_settings
from focused_research_agent.ui.exceptions import BackendUnavailableError

_HEALTH_ENDPOINT = "/health"
_RESEARCH_ENDPOINT = "/api/v1/research"
_CHAT_ENDPOINT = "/api/v1/chat"
_CONVERSATIONS_ENDPOINT = "/api/v1/conversations"
_REPORT_ENDPOINT = "/api/v1/report"
_REPORTS_ENDPOINT = "/api/v1/reports"

_TIMEOUT_ERROR_MESSAGE = "Request timed out — research is taking too long."


class ResearchCallResult(TypedDict):
    success: bool
    data: dict | None
    error: str | None


def check_health() -> bool:
    """
    Check whether the FastAPI backend is reachable.

    Makes a GET request to the /health endpoint with a short fixed timeout.
    A failed health check is not an error — it means the backend is offline.
    This function never raises; it always returns a bool.

    Returns:
        bool: True if the backend responded with HTTP 200, False otherwise.
    """
    settings = get_ui_settings()
    try:
        response = httpx.get(f"{settings.api_base_url}{_HEALTH_ENDPOINT}", timeout=5.0)
        return response.status_code == 200
    except httpx.ConnectError:
        return False


def _parse_post_response(response: httpx.Response) -> ResearchCallResult:
    """
    Parse an httpx response from a POST research or chat request
    into a ResearchCallResult.

    Handles HTTP 200, 400, 422, and any other status code. Does not
    handle connection or timeout exceptions — those are caught by the
    calling function.

    Args:
        response: The httpx response object from the POST request.

    Returns:
        ResearchCallResult: Parsed result with success, data, and
            error fields populated based on the response status code.
    """
    if response.status_code == 200:
        return {"success": True, "data": response.json(), "error": None}

    if response.status_code == 400:
        return {"success": False, "data": None, "error": response.json()["detail"]}

    if response.status_code == 422:
        return {"success": False, "data": None, "error": "Invalid question submitted."}

    return {
        "success": False,
        "data": None,
        "error": f"Unexpected error: {response.status_code}",
    }


def call_research(question: str) -> ResearchCallResult:
    """
    Send a research question to the FastAPI backend and return the result.

    Makes a POST request to the versioned research endpoint with the user's
    question as the JSON body. Always returns a ResearchCallResult with three
    keys: success, data, and error. The shape is consistent across all
    response paths so that 1_🔍_Research.py and views.py never have to guess what
    they are receiving.

    Args:
        question: The user's research question to send to the backend.

    Returns:
        ResearchCallResult: A typed dict with the following keys:
            - success (bool): True if the backend returned HTTP 200,
                False for all other responses.
            - data (dict | None): The full research response from the
                backend when success is True, otherwise None.
            - error (str | None): A human-readable error message when
                success is False, otherwise None.

    Raises:
        BackendUnavailableError: If the backend cannot be reached at
            the configured UI_API_BASE_URL. Raised instead of returning
            an error dict because a completely unreachable backend is a
            different category of failure from a bad response — it means
            the user needs to start the backend before trying again.
    """
    settings = get_ui_settings()
    try:
        response = httpx.post(
            f"{settings.api_base_url}{_RESEARCH_ENDPOINT}",
            json={"question": question},
            timeout=settings.request_timeout,
        )
        return _parse_post_response(response)
    except httpx.ConnectError:
        raise BackendUnavailableError(
            f"Cannot connect to backend at {settings.api_base_url} — is FastAPI running?"
        )
    except httpx.TimeoutException:
        return {
            "success": False,
            "data": None,
            "error": _TIMEOUT_ERROR_MESSAGE,
        }


def call_chat(question: str, conversation_id: str | None) -> ResearchCallResult:
    """
    Send a chat turn to the FastAPI backend and return the result.

    Makes a POST request to the chat endpoint with the user's question
    and optional conversation ID. Returns a ResearchCallResult with the
    same shape as call_research, but the data dict additionally contains
    conversation_id and turn_number fields.

    Args:
        question: The user's research question for this turn.
        conversation_id: Existing conversation UUID to continue, or
            None to start a new conversation.

    Returns:
        ResearchCallResult: A typed dict with success, data, and error
            keys. On success, data contains the full chat response
            including conversation_id and turn_number.

    Raises:
        BackendUnavailableError: If the backend cannot be reached.
    """
    settings = get_ui_settings()
    try:
        response = httpx.post(
            f"{settings.api_base_url}{_CHAT_ENDPOINT}",
            json={"question": question, "conversation_id": conversation_id},
            timeout=settings.request_timeout,
        )
        return _parse_post_response(response)
    except httpx.ConnectError:
        raise BackendUnavailableError(
            f"Cannot connect to backend at {settings.api_base_url} — is FastAPI running?"
        )
    except httpx.TimeoutException:
        return {
            "success": False,
            "data": None,
            "error": _TIMEOUT_ERROR_MESSAGE,
        }


def call_report(question: str) -> ResearchCallResult:
    """
    Send a report generation request to the FastAPI backend and
    return the result.

    Makes a POST request to the report endpoint with the user's
    question. Returns a ResearchCallResult with the same shape as
    call_research, but the answer field contains a structured
    markdown report with Introduction, Key Findings, Analysis,
    and Conclusion sections.

    Args:
        question: The user's research question for the report.

    Returns:
        ResearchCallResult: A typed dict with success, data, and
            error keys. On success, data contains the full report
            response with structured markdown in the answer field.

    Raises:
        BackendUnavailableError: If the backend cannot be reached.
    """

    settings = get_ui_settings()
    try:
        response = httpx.post(
            f"{settings.api_base_url}{_REPORT_ENDPOINT}",
            json={"question": question},
            timeout=settings.request_timeout,
        )
        return _parse_post_response(response)
    except httpx.ConnectError:
        raise BackendUnavailableError(
            f"Cannot connect to backend at {settings.api_base_url} — is FastAPI running?"
        )
    except httpx.TimeoutException:
        return {
            "success": False,
            "data": None,
            "error": _TIMEOUT_ERROR_MESSAGE,
        }


def get_conversations() -> list[dict]:
    """
    Fetch the list of all past conversations from the backend.

    Makes a GET request to the conversations endpoint. Returns an
    empty list on any error so that history panel failures never
    block the chat UI from functioning.

    Returns:
        list[dict]: List of conversation summary dicts containing
            conversation_id, title, and created_at keys.
            Empty list if the request fails for any reason.
    """
    settings = get_ui_settings()
    try:
        response = httpx.get(
            f"{settings.api_base_url}{_CONVERSATIONS_ENDPOINT}",
            timeout=settings.request_timeout,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return []
    except httpx.ConnectError:
        return []

    except httpx.TimeoutException:
        return []


def get_conversation(conversation_id: str) -> list[dict]:
    """
    Fetch all turns of a specific conversation from the backend.

    Makes a GET request to the conversation detail endpoint. Returns
    an empty list on any error so that history loading failures never
    block the chat UI.

    Args:
        conversation_id: UUID string identifying the conversation to
            fetch.

    Returns:
        list[dict]: List of complete turn dicts in chronological order.
            Empty list if the request fails for any reason.
    """
    settings = get_ui_settings()
    try:
        response = httpx.get(
            f"{settings.api_base_url}{_CONVERSATIONS_ENDPOINT}/{conversation_id}",
            timeout=settings.request_timeout,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return []
    except httpx.ConnectError:
        return []

    except httpx.TimeoutException:
        return []


def get_reports() -> list[dict]:
    """
    Fetch the list of all past report runs from the backend.

    Returns an empty list on any error so sidebar failures never
    block the report UI.

    Returns:
        list[dict]: List of report summary dicts containing
            conversation_id, title, and created_at keys.
            Empty list if the request fails for any reason.
    """
    settings = get_ui_settings()
    try:
        response = httpx.get(
            f"{settings.api_base_url}{_REPORTS_ENDPOINT}",
            timeout=settings.request_timeout,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except httpx.ConnectError:
        return []
    except httpx.TimeoutException:
        return []
