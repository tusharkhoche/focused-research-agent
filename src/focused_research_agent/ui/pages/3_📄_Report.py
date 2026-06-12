"""
Streamlit report page for the Focused Research Agent UI.

This module implements the deep research report generation interface.
It uses Streamlit's multi-page convention — placing it in the pages/
folder with a 3_ prefix makes it appear third in the sidebar navigation.

The report page uses advanced Tavily search depth and a structured
prompt that produces a markdown report with Introduction, Key Findings,
Analysis, and Conclusion sections. Report generation takes longer than
quick research — users are informed of this via a caption and spinner.

Architecturally, this module is a UI transport entrypoint alongside
Home.py and the other pages. It follows the same thin wiring pattern.
"""

import streamlit as st
from focused_research_agent.ui.exceptions import BackendUnavailableError
from focused_research_agent.ui.views import render_health_status
from focused_research_agent.ui.api_client import (
    call_report,
    check_health,
    get_conversation,
    get_reports,
)


def _init_session_state() -> None:
    """
    Initialise session state keys for the report page.
    ...
    """
    if "report_result" not in st.session_state:
        st.session_state.report_result = None

    if "report_question" not in st.session_state:
        st.session_state.report_question = ""

    if "report_generating" not in st.session_state:
        st.session_state.report_generating = False


def _render_sidebar() -> None:
    """
    Render sidebar content for the report page.

    Displays the page title, API health status, and a list of
    past report runs with load buttons.

    Returns:
        None
    """
    st.sidebar.title("📄 Report")
    render_health_status(check_health())

    reports = get_reports()
    if reports:
        st.sidebar.subheader("📋 Past Reports")
        for report in reports:
            with st.sidebar.expander(report["title"] or "Untitled"):
                if st.button("Load", key=report["conversation_id"]):
                    turns = get_conversation(report["conversation_id"])
                    if turns:
                        st.session_state.report_result = {
                            "success": True,
                            "data": turns[0],
                        }
                    st.rerun()


def _render_report_input() -> str | None:
    question = st.text_area(
        "What would you like a full report on?",
        height=100,
        placeholder="e.g. The impact of quantum computing on Artificial Intelligence",
    )
    if st.button("📄 Generate Report"):
        return question
    return None


def _render_report_success(data: dict) -> None:
    """
    Render the successful report content.

    Args:
        data: The full report response dict from the backend.

    Returns:
        None
    """
    st.success("✅ Report generated successfully!")
    st.divider()
    st.markdown(data["answer"])
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Queries", len(data.get("queries") or []))
    with col2:
        st.metric("🔗 Sources", len(data.get("sources") or []))
    with col3:
        st.metric("✅ Citations", len(data.get("citations") or []))

    st.divider()

    images = data.get("images") or []
    if images:
        st.subheader("🖼️ Images")
        cols = st.columns(min(len(images), 3))
        for index, url in enumerate(images):
            with cols[index % 3]:
                try:
                    st.image(url, use_container_width=True)
                except Exception:
                    pass
        st.divider()

    if data.get("sources"):
        st.subheader("📚 Sources")
        for source in data["sources"]:
            with st.expander(source["title"]):
                st.write(source["url"])
                st.caption(source["snippet"])

    st.divider()

    if st.checkbox("🛠️ Show raw response"):
        st.json(data)


def _render_report_result() -> None:
    """
    Render the most recent report result from session state.

    Returns:
        None
    """
    if st.session_state.report_result is None:
        return

    result = st.session_state.report_result

    # Handle transport-level failures — 422, 500, connection error
    # In these cases result["data"] is None
    if not result["success"] and result["data"] is None:  # ← add this block
        st.error(result["error"] or "An error occurred.")
        return

    data = result["data"]

    if data.get("status") == "error" or data.get("answer") is None:
        errors = data.get("errors") or ["An unknown error occurred."]
        st.error(f"Research failed: {errors[0]}")
        if st.checkbox("🛠️ Show raw response"):
            st.json(result)
        return

    if result["success"]:
        _render_report_success(data)
    else:
        st.error(result["error"] or "An error occurred.")


st.set_page_config(page_title="Research Report", layout="centered")
st.title("📄 Research Report")
st.caption("Deep research with structured analysis — takes longer than quick research.")

_init_session_state()
_render_sidebar()

question = _render_report_input()

if question is not None:
    try:
        with st.spinner("Generating report — this may take a minute..."):
            result = call_report(question)
        st.session_state.report_result = result
    except BackendUnavailableError as e:
        st.error(str(e))
        st.stop()

_render_report_result()
