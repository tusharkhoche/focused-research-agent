"""
Tests for the application-layer chat use case.

What is tested:
- execute_chat_turn raises ApplicationError for invalid questions
- execute_chat_turn generates a new conversation_id when none provided
- execute_chat_turn reuses the provided conversation_id on follow-up
- execute_chat_turn returns result with conversation_id and turn_number
- execute_chat_turn sets turn_number to 1 for first turn
- execute_chat_turn sets turn_number correctly for follow-up turns
- execute_chat_turn passes conversation_history to the graph state
- execute_chat_turn returns result even when save_run fails
- _build_chat_initial_state populates conversation fields correctly

How it is tested:
- An in-memory SQLite database is used for full integration-style
  tests so repository calls work correctly without mocking
- build_graph is patched with a fake graph so no real LLM or search
  calls are made
- get_conversation_history is tested indirectly through the full
  execute_chat_turn flow

Why it matters:
- Verifies the application layer correctly threads conversation
  context between the database and the graph
- Confirms that persistence failure does not break the research result
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

import focused_research_agent.application.chat_use_case as chat_use_case_module
from focused_research_agent.application.chat_use_case import (
    _build_chat_initial_state,
    execute_chat_turn,
)
from focused_research_agent.application.exceptions import ApplicationError
from focused_research_agent.database.models import Base
from focused_research_agent.database.repository import save_run

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> Session:
    """
    Create a fresh in-memory SQLite database and session for each test.

    Yields:
        Session: An active SQLAlchemy session connected to the
            in-memory database.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def completed_state() -> dict:
    """
    Return a realistic completed research state for seeding the database.

    Returns:
        dict: A state dict matching the normalized research result shape.
    """
    return {
        "run_id": "run-seed-123",
        "question": "What is quantum computing?",
        "status": "completed",
        "scope": "Explain quantum computing clearly",
        "queries": ["quantum computing overview"],
        "sources": [
            {
                "title": "Quantum Overview",
                "url": "https://example.com/quantum",
                "snippet": "Quantum computing uses quantum mechanics.",
                "source": "tavily",
                "score": 0.95,
            }
        ],
        "answer": "Quantum computing uses quantum mechanical phenomena.",
        "citations": ["https://example.com/quantum"],
        "errors": [],
    }


class FakeGraph:
    """
    Fake LangGraph graph that returns a deterministic completed result
    without making real LLM or search provider calls.
    """

    def invoke(self, initial_state: dict) -> dict:
        """
        Return a fixed successful graph result.

        Args:
            initial_state: Initial graph state passed into the workflow.

        Returns:
            dict: Mocked completed graph result.
        """
        return {
            "run_id": "run-fake-456",
            "question": initial_state["question"],
            "status": "completed",
            "scope": "Explain the topic clearly",
            "assumptions": ["User is a beginner"],
            "constraints": {},
            "queries": ["query one", "query two", "query three"],
            "sources": [
                {
                    "title": "Test Source",
                    "url": "https://example.com/source",
                    "snippet": "A test source snippet.",
                    "source": "mock",
                    "score": 0.9,
                }
            ],
            "answer": "This is a synthesized answer.",
            "citations": ["https://example.com/source"],
            "errors": [],
            "debug": None,
            "conversation_id": initial_state.get("conversation_id"),
            "conversation_history": initial_state.get("conversation_history"),
        }


def fake_build_graph():
    """Return a FakeGraph instance for testing."""
    return FakeGraph()


# ---------------------------------------------------------------------------
# _build_chat_initial_state tests
# ---------------------------------------------------------------------------


def test_build_chat_initial_state_sets_conversation_id():
    state = _build_chat_initial_state(
        question="What is AI?",
        conversation_id="conv-abc",
        conversation_history=None,
    )

    assert state["conversation_id"] == "conv-abc"


def test_build_chat_initial_state_sets_conversation_history():
    history = [{"turn": 1, "question": "Q1", "answer": "A1", "scope": "S1"}]

    state = _build_chat_initial_state(
        question="What is AI?",
        conversation_id="conv-abc",
        conversation_history=history,
    )

    assert state["conversation_history"] == history


def test_build_chat_initial_state_sets_none_history():
    state = _build_chat_initial_state(
        question="What is AI?",
        conversation_id="conv-abc",
        conversation_history=None,
    )

    assert state["conversation_history"] is None


def test_build_chat_initial_state_sets_question():
    state = _build_chat_initial_state(
        question="What is AI?",
        conversation_id="conv-abc",
        conversation_history=None,
    )

    assert state["question"] == "What is AI?"


# ---------------------------------------------------------------------------
# execute_chat_turn validation tests
# ---------------------------------------------------------------------------


def test_execute_chat_turn_raises_for_empty_question(db, monkeypatch):
    monkeypatch.setattr(chat_use_case_module, "build_graph", fake_build_graph)

    with pytest.raises(ApplicationError, match="No user query provided"):
        execute_chat_turn(db=db, conversation_id=None, question="   ")


def test_execute_chat_turn_raises_for_non_string_question(db, monkeypatch):
    monkeypatch.setattr(chat_use_case_module, "build_graph", fake_build_graph)

    with pytest.raises(ApplicationError, match="User query must be a string"):
        execute_chat_turn(db=db, conversation_id=None, question=123)  # type: ignore


def test_execute_chat_turn_raises_for_punctuation_only_question(db, monkeypatch):
    monkeypatch.setattr(chat_use_case_module, "build_graph", fake_build_graph)

    with pytest.raises(ApplicationError):
        execute_chat_turn(db=db, conversation_id=None, question="...")


# ---------------------------------------------------------------------------
# execute_chat_turn conversation ID tests
# ---------------------------------------------------------------------------


def test_execute_chat_turn_generates_conversation_id_when_none(db, monkeypatch):
    monkeypatch.setattr(chat_use_case_module, "build_graph", fake_build_graph)

    result = execute_chat_turn(db=db, conversation_id=None, question="What is AI?")

    assert result["conversation_id"] is not None
    assert len(result["conversation_id"]) > 0


def test_execute_chat_turn_reuses_provided_conversation_id(db, monkeypatch):
    monkeypatch.setattr(chat_use_case_module, "build_graph", fake_build_graph)

    result = execute_chat_turn(
        db=db,
        conversation_id="conv-existing-123",
        question="What is AI?",
    )

    assert result["conversation_id"] == "conv-existing-123"


def test_execute_chat_turn_two_calls_with_none_generate_different_ids(db, monkeypatch):
    monkeypatch.setattr(chat_use_case_module, "build_graph", fake_build_graph)

    result1 = execute_chat_turn(db=db, conversation_id=None, question="What is AI?")
    result2 = execute_chat_turn(db=db, conversation_id=None, question="What is ML?")

    assert result1["conversation_id"] != result2["conversation_id"]


# ---------------------------------------------------------------------------
# execute_chat_turn turn number tests
# ---------------------------------------------------------------------------


def test_execute_chat_turn_sets_turn_number_to_1_for_first_turn(db, monkeypatch):
    monkeypatch.setattr(chat_use_case_module, "build_graph", fake_build_graph)

    result = execute_chat_turn(db=db, conversation_id=None, question="What is AI?")

    assert result["turn_number"] == 1


def test_execute_chat_turn_sets_turn_number_to_2_for_follow_up(
    db, monkeypatch, completed_state
):
    monkeypatch.setattr(chat_use_case_module, "build_graph", fake_build_graph)

    save_run(db, completed_state, conversation_id="conv-abc", turn_number=1)

    result = execute_chat_turn(
        db=db,
        conversation_id="conv-abc",
        question="What are its limitations?",
    )

    assert result["turn_number"] == 2


# ---------------------------------------------------------------------------
# execute_chat_turn result shape tests
# ---------------------------------------------------------------------------


def test_execute_chat_turn_returns_expected_result_shape(db, monkeypatch):
    monkeypatch.setattr(chat_use_case_module, "build_graph", fake_build_graph)

    result = execute_chat_turn(db=db, conversation_id=None, question="What is AI?")

    assert "run_id" in result
    assert "question" in result
    assert "status" in result
    assert "answer" in result
    assert "conversation_id" in result
    assert "turn_number" in result


def test_execute_chat_turn_persists_run_to_database(db, monkeypatch):
    monkeypatch.setattr(chat_use_case_module, "build_graph", fake_build_graph)

    result = execute_chat_turn(db=db, conversation_id=None, question="What is AI?")

    from focused_research_agent.database.repository import get_conversation_turns

    turns = get_conversation_turns(db, result["conversation_id"])

    assert len(turns) == 1
    assert turns[0]["question"] == "What is AI?"


def test_execute_chat_turn_returns_result_even_when_save_fails(db, monkeypatch):
    monkeypatch.setattr(chat_use_case_module, "build_graph", fake_build_graph)

    from sqlalchemy.exc import SQLAlchemyError  # ← add import

    def fake_save_run(*args, **kwargs):
        raise SQLAlchemyError(
            "Database write failed"
        )  # ← change RuntimeError to SQLAlchemyError

    monkeypatch.setattr(chat_use_case_module, "save_run", fake_save_run)

    result = execute_chat_turn(db=db, conversation_id=None, question="What is AI?")

    assert result["answer"] is not None
    assert result["status"] == "completed"
