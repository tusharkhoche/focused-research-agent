"""
Streamlit application entrypoint for the Focused Research Agent UI.

This module wires together the api_client and views layers. It manages
session state, reads user input, calls the backend through api_client,
and delegates all rendering to views.

Architecturally, this module is the UI transport entrypoint — the same
role cli.py plays for the terminal transport. It should contain no HTTP
logic and no direct st.* rendering beyond layout and input widgets.
"""

import streamlit as st
from focused_research_agent.ui.api_client import call_research, check_health
from focused_research_agent.ui.views import (
    render_research_details,
    render_error,
    render_sources,
    render_answer,
    render_health_status,
    render_metrics,
)
from focused_research_agent.ui.exceptions import BackendUnavailableError


def _init_session_state() -> None:
    """
    Initialise session state keys used across reruns.

    Uses the guard pattern to avoid resetting state on every rerun.
    Must be called once at the top of the script before any widgets
    that depend on session state are rendered.

    Returns:
        None
    """
    if "history" not in st.session_state:
        st.session_state.history = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def _render_sidebar() -> None:
    """
    Render all sidebar content including settings title, health status,
    and past research session history.

    Returns:
        None
    """
    st.sidebar.title("⚙️ Settings")
    is_online = check_health()
    render_health_status(is_online)

    if st.session_state.history:
        st.sidebar.subheader("📋 History")
        for item in reversed(st.session_state.history):
            with st.sidebar.expander(item["question"][:60]):
                st.write(item["answer"])
                st.caption(f"Run ID: {item['run_id']}")


def _render_input() -> str:
    """
    Render the research question input area and return the current value.

    Returns:
        str: The current text entered by the user. Empty string if nothing
            entered yet.
    """
    user_query = st.text_area(
        "What would you like to research?",
        placeholder="e.g. What are the latest advances in quantum computing?",
        height=100,
    )
    return user_query


def _handle_research(question: str) -> None:
    """
    Render the research button and handle the research request.

    Calls the backend when the button is clicked and a valid question
    is provided. Stores the result in session state so it persists
    across reruns. Stops the script immediately if the backend is
    unreachable.

    Args:
        question: The current value from the question input area.

    Returns:
        None
    """
    if st.button("🔍 Research"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("Researching... this may take up to 2 minutes."):
                    result = call_research(question)
                    st.session_state.last_result = result
                if result["success"]:
                    st.session_state.history.append(
                        {
                            "question": question,
                            "answer": result["data"]["answer"],
                            "run_id": result["data"]["run_id"],
                        }
                    )
            except BackendUnavailableError as e:
                st.error(str(e))
                st.stop()


def _render_results() -> None:
    """
    Render the most recent research result from session state.

    Reads last_result from session state and delegates rendering to
    views. Renders nothing if no result exists yet.

    Returns:
        None
    """
    if st.session_state.last_result is not None:
        result = st.session_state.last_result

        if result["success"]:
            render_answer(result["data"])
            render_metrics(result["data"])
            render_research_details(result["data"])
            render_sources(result["data"]["sources"], result["data"].get("images"))
        else:
            render_error(result["error"])

        st.divider()
        if st.checkbox("🛠️ Show raw response"):
            st.json(result)


st.set_page_config(page_title="Focused Research Agent", layout="centered")
st.title("🔍 Focused Research Agent")

_init_session_state()
_render_sidebar()
question = _render_input()
_handle_research(question)
_render_results()
