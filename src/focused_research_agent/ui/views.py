"""
Streamlit rendering functions for the Focused Research Agent UI.

This module is the only file in the UI layer that imports streamlit.
It takes plain Python data and renders Streamlit widgets. It contains
no HTTP logic and no calls to api_client.

Architecturally, this module is the presentation layer of the UI
transport — the same role the format_* functions play in cli.py, but
rendering widgets instead of building terminal strings.
"""

import streamlit as st


def render_health_status(is_online: bool) -> None:
    """
    Render the backend health status in the sidebar.

    Displays a green success banner when the backend is reachable and a
    red error banner when it is not. Called on every page load so the
    user always knows whether the backend is running.

    Args:
        is_online: True if the backend responded with HTTP 200,
            False otherwise.

    Returns:
        None
    """
    if is_online:
        st.sidebar.success("✅ API Online")
    else:
        st.sidebar.error("❌ API Offline — start FastAPI first")


def render_error(message: str) -> None:
    """
    Render a user-facing error message in the main area.

    Displays a red error banner with the provided message. Called when
    the research request fails for any reason other than the backend
    being completely unreachable.

    Args:
        message: Human-readable error message to display.

    Returns:
        None
    """
    st.error(message)


def render_answer(data: dict) -> None:
    """
    Render the research answer in the main area.

    Displays a success banner followed by the synthesized answer text.
    Called only when the research request succeeds.

    Args:
        data: The full research response dict returned by the backend.
            Expected to contain at minimum an "answer" key.

    Returns:
        None
    """
    st.success("✅ Research complete!")
    st.markdown(data["answer"])
    st.divider()


def render_research_details(data: dict) -> None:
    """
    Render research details in a collapsible expander.

    Displays scope, queries, citations, and run ID inside a collapsed
    expander so these details are available without dominating the page.
    Each section only renders if the backend returned data for it.

    Args:
        data: The full research response dict returned by the backend.

    Returns:
        None
    """
    with st.expander("🔍 Research Details", expanded=False):
        if data["scope"] is not None:
            st.subheader("Scope")
            st.markdown(data["scope"])

        if data["queries"] is not None:
            st.subheader("Queries")
            for query in data["queries"]:
                st.write(f"- {query}")

        if data["citations"] is not None:
            st.subheader("Citations")
            for citation in data["citations"]:
                st.write(citation)
        st.caption(f"Run ID: {data['run_id']}")
    st.divider()


def _extract_image_urls(sources: list[dict]) -> list[str]:
    """
    Extract image URLs from a list of source dicts.

    Scans source URLs for known image file extensions and returns
    any that match. Used to render a visual images section below
    the research answer.

    Args:
        sources: List of source dicts from the backend response.

    Returns:
        list[str]: List of image URLs found in the sources.
            Empty list if no image URLs are found.
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    image_urls = []

    for source in sources:
        url = source.get("url", "").lower()
        if url.endswith(image_extensions):
            image_urls.append(source["url"])

    return image_urls


def render_sources(sources: list[dict], images: list[str] | None = None) -> None:
    """
    Render the list of research sources as collapsible expanders.
    Renders image URLs above the sources when available.

    Args:
        sources: List of source dicts returned by the backend.
        images: Optional list of image URLs from the search results.

    Returns:
        None
    """

    if images:
        st.subheader("🖼️ Images")
        cols = st.columns(min(len(images), 3))
        for index, url in enumerate(images):
            with cols[index % 3]:
                try:
                    st.image(url, use_container_width=True)
                except Exception:
                    pass

    st.subheader("📚 Sources")
    if not sources:
        st.info("No sources available.")
        return

    for source in sources:
        with st.expander(source["title"]):
            st.write(source["url"])
            st.caption(source["snippet"])


def render_metrics(data: dict) -> None:
    """
    Render a summary metrics row showing research run statistics.

    Displays query count, source count, and citation count as metric
    widgets in a three-column layout. Gives the user an immediate
    sense of research depth before reading the answer.

    Args:
        data: The full research response dict returned by the backend.

    Returns:
        None
    """
    # | 📋 5 Queries | 🔗 8 Sources | ✅ 3 Citations |
    if data["queries"] is not None:
        queries_count = len(data["queries"])
    else:
        queries_count = 0

    if data["sources"] is not None:
        sources_count = len(data["sources"])
    else:
        sources_count = 0

    if data["citations"] is not None:
        citations_count = len(data["citations"])
    else:
        citations_count = 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📋 Queries", queries_count)

    with col2:
        st.metric("🔗 Sources", sources_count)

    with col3:
        st.metric("✅ Citations", citations_count)

    st.divider()
