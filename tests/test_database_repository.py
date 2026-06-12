"""
Tests for the database repository layer.

What is tested:
- _serialize and _deserialize helper functions
- save_run creates a correct ConversationRun row
- save_run sets conversation_title only on turn 1
- get_conversation_history returns turns in chronological order
- get_conversation_history respects the max_turns limit
- get_conversation_history returns empty list for unknown conversation
- get_all_conversations returns one entry per conversation
- get_all_conversations returns newest first
- get_all_conversations returns empty list when no data exists
- get_conversation_turns returns all turns with deserialized fields
- get_conversation_turns returns empty list for unknown conversation

How it is tested:
- An in-memory SQLite database is created fresh for each test using
  pytest fixtures. This avoids any dependency on a real database file
  and guarantees test isolation — each test starts with a clean slate.

Why it matters:
- Verifies that serialization, deserialization, and all CRUD
  operations work correctly before the application layer depends
  on them.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from focused_research_agent.database.models import Base, ConversationRun
from focused_research_agent.database.repository import (
    _deserialize,
    _serialize,
    get_all_conversations,
    get_conversation_history,
    get_conversation_turns,
    save_run,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> Session:
    """
    Create a fresh in-memory SQLite database and session for each test.

    Uses SQLite in-memory mode (sqlite:///:memory:) so no files are
    created on disk and the database is automatically destroyed when
    the test finishes. Creates all tables before the test and yields
    a session ready to use.

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
def sample_state() -> dict:
    """
    Return a realistic normalized research state dict for testing.

    Returns:
        dict: A state dict matching the shape returned by the
            application layer's normalize_state function.
    """
    return {
        "run_id": "run-test-123",
        "question": "What is quantum computing?",
        "status": "completed",
        "scope": "Explain quantum computing clearly",
        "queries": [
            "quantum computing overview",
            "quantum computing applications",
            "quantum computing limitations",
        ],
        "sources": [
            {
                "title": "Quantum Computing Overview",
                "url": "https://example.com/quantum",
                "snippet": "Quantum computing uses quantum mechanics.",
                "source": "tavily",
                "score": 0.95,
            }
        ],
        "answer": "Quantum computing uses quantum mechanical phenomena.",
        "citations": ["https://example.com/quantum"],
        "errors": [],
        "images": [  # ← add this
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
        ],
    }


@pytest.fixture
def sample_followup_state() -> dict:
    """
    Return a realistic follow-up research state dict for testing
    multi-turn conversations.

    Returns:
        dict: A state dict for a follow-up question in a conversation.
    """
    return {
        "run_id": "run-test-456",
        "question": "What are its limitations?",
        "status": "completed",
        "scope": "Explain quantum computing limitations",
        "queries": [
            "quantum computing limitations",
            "quantum computing challenges",
        ],
        "sources": [
            {
                "title": "Quantum Limitations",
                "url": "https://example.com/limits",
                "snippet": "Quantum computers face decoherence issues.",
                "source": "tavily",
                "score": 0.91,
            }
        ],
        "answer": "Quantum computing faces challenges like decoherence.",
        "citations": ["https://example.com/limits"],
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Serialization helper tests
# ---------------------------------------------------------------------------


def test_serialize_returns_json_string_for_list():
    result = _serialize(["query one", "query two"])

    assert result == '["query one", "query two"]'


def test_serialize_returns_none_for_none():
    result = _serialize(None)

    assert result is None


def test_deserialize_returns_list_for_json_string():
    result = _deserialize('["query one", "query two"]')

    assert result == ["query one", "query two"]


def test_deserialize_returns_none_for_none():
    result = _deserialize(None)

    assert result is None


def test_serialize_then_deserialize_roundtrip():
    original = ["query one", "query two", "query three"]

    assert _deserialize(_serialize(original)) == original


# ---------------------------------------------------------------------------
# save_run tests
# ---------------------------------------------------------------------------


def test_save_run_creates_row_in_database(db, sample_state):
    save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )

    count = db.query(ConversationRun).count()

    assert count == 1


def test_save_run_returns_conversation_run_with_id(db, sample_state):
    result = save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )

    assert isinstance(result, ConversationRun)
    assert result.id is not None


def test_save_run_stores_correct_field_values(db, sample_state):
    result = save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )

    assert result.run_id == "run-test-123"
    assert result.question == "What is quantum computing?"
    assert result.status == "completed"
    assert result.scope == "Explain quantum computing clearly"
    assert result.answer == "Quantum computing uses quantum mechanical phenomena."
    assert result.conversation_id == "conv-abc"
    assert result.turn_number == 1


def test_save_run_sets_conversation_title_on_turn_1(db, sample_state):
    result = save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )

    assert result.conversation_title == "What is quantum computing?"


def test_save_run_does_not_set_conversation_title_after_turn_1(
    db, sample_followup_state
):
    result = save_run(
        db,
        sample_followup_state,
        conversation_id="conv-abc",
        turn_number=2,
        mode="research",
    )

    assert result.conversation_title is None


def test_save_run_truncates_title_to_60_characters(db):
    long_question_state = {
        "run_id": "run-long",
        "question": "A" * 100,
        "status": "completed",
        "scope": None,
        "queries": None,
        "sources": None,
        "answer": None,
        "citations": None,
        "errors": [],
    }

    result = save_run(
        db,
        long_question_state,
        conversation_id="conv-abc",
        turn_number=1,
        mode="research",
    )

    assert len(result.conversation_title) == 60


def test_save_run_serializes_list_fields(db, sample_state):
    result = save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )

    assert (
        result.queries
        == '["quantum computing overview", "quantum computing applications", "quantum computing limitations"]'
    )
    assert result.citations == '["https://example.com/quantum"]'
    assert result.errors == "[]"


def test_save_run_handles_none_list_fields(db):
    minimal_state = {
        "run_id": "run-minimal",
        "question": "What is AI?",
        "status": "error",
        "scope": None,
        "queries": None,
        "sources": None,
        "answer": None,
        "citations": None,
        "errors": ["init_run: No question provided"],
    }

    result = save_run(
        db, minimal_state, conversation_id="conv-xyz", turn_number=1, mode="research"
    )

    assert result.queries is None
    assert result.sources is None
    assert result.citations is None


def test_save_run_sets_created_at_and_updated_at(db, sample_state):
    result = save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )

    assert result.created_at is not None
    assert result.updated_at is not None


# ---------------------------------------------------------------------------
# get_conversation_history tests
# ---------------------------------------------------------------------------


def test_get_conversation_history_returns_turns_in_chronological_order(
    db, sample_state, sample_followup_state
):
    save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )
    save_run(
        db,
        sample_followup_state,
        conversation_id="conv-abc",
        turn_number=2,
        mode="research",
    )

    history = get_conversation_history(db, conversation_id="conv-abc", max_turns=5)

    assert len(history) == 2
    assert history[0]["turn"] == 1
    assert history[1]["turn"] == 2


def test_get_conversation_history_returns_correct_fields(db, sample_state):
    save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )

    history = get_conversation_history(db, conversation_id="conv-abc", max_turns=5)

    assert history[0]["question"] == "What is quantum computing?"
    assert (
        history[0]["answer"] == "Quantum computing uses quantum mechanical phenomena."
    )
    assert history[0]["scope"] == "Explain quantum computing clearly"


def test_get_conversation_history_respects_max_turns_limit(
    db, sample_state, sample_followup_state
):
    save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )
    save_run(
        db,
        sample_followup_state,
        conversation_id="conv-abc",
        turn_number=2,
        mode="research",
    )

    history = get_conversation_history(db, conversation_id="conv-abc", max_turns=1)

    assert len(history) == 1
    assert history[0]["turn"] == 2


def test_get_conversation_history_returns_empty_list_for_unknown_conversation(
    db,
):
    history = get_conversation_history(
        db, conversation_id="conv-does-not-exist", max_turns=5
    )

    assert history == []


def test_get_conversation_history_only_returns_turns_for_given_conversation(
    db, sample_state, sample_followup_state
):
    save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )
    save_run(
        db,
        sample_followup_state,
        conversation_id="conv-xyz",
        turn_number=1,
        mode="research",
    )

    history = get_conversation_history(db, conversation_id="conv-abc", max_turns=5)

    assert len(history) == 1
    assert history[0]["question"] == "What is quantum computing?"


# ---------------------------------------------------------------------------
# get_all_conversations tests
# ---------------------------------------------------------------------------


def test_get_all_conversations_returns_empty_list_when_no_data(db):
    result = get_all_conversations(db)

    assert result == []


def test_get_all_conversations_returns_one_entry_per_conversation(
    db, sample_state, sample_followup_state
):
    save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )
    save_run(
        db,
        sample_followup_state,
        conversation_id="conv-abc",
        turn_number=2,
        mode="research",
    )
    save_run(
        db, sample_state, conversation_id="conv-xyz", turn_number=1, mode="research"
    )

    result = get_all_conversations(db)

    assert len(result) == 2


def test_get_all_conversations_returns_correct_fields(db, sample_state):
    save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )

    result = get_all_conversations(db)

    assert result[0]["conversation_id"] == "conv-abc"
    assert result[0]["title"] == "What is quantum computing?"
    assert "created_at" in result[0]


def test_get_all_conversations_returns_newest_first(
    db, sample_state, sample_followup_state
):
    save_run(
        db, sample_state, conversation_id="conv-first", turn_number=1, mode="research"
    )
    save_run(
        db,
        sample_followup_state,
        conversation_id="conv-second",
        turn_number=1,
        mode="research",
    )

    result = get_all_conversations(db)

    assert result[0]["conversation_id"] == "conv-second"
    assert result[1]["conversation_id"] == "conv-first"


# ---------------------------------------------------------------------------
# get_conversation_turns tests
# ---------------------------------------------------------------------------


def test_get_conversation_turns_returns_empty_list_for_unknown_conversation(
    db,
):
    result = get_conversation_turns(db, conversation_id="conv-unknown")

    assert result == []


def test_get_conversation_turns_returns_all_turns_in_order(
    db, sample_state, sample_followup_state
):
    save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )
    save_run(
        db,
        sample_followup_state,
        conversation_id="conv-abc",
        turn_number=2,
        mode="research",
    )

    result = get_conversation_turns(db, conversation_id="conv-abc")

    assert len(result) == 2
    assert result[0]["turn_number"] == 1
    assert result[1]["turn_number"] == 2


def test_get_conversation_turns_deserializes_list_fields(db, sample_state):
    save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )

    result = get_conversation_turns(db, conversation_id="conv-abc")

    assert result[0]["queries"] == [
        "quantum computing overview",
        "quantum computing applications",
        "quantum computing limitations",
    ]
    assert result[0]["citations"] == ["https://example.com/quantum"]
    assert result[0]["errors"] == []


def test_get_conversation_turns_returns_correct_fields(db, sample_state):
    save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="research"
    )

    result = get_conversation_turns(db, conversation_id="conv-abc")

    assert result[0]["run_id"] == "run-test-123"
    assert result[0]["question"] == "What is quantum computing?"
    assert result[0]["status"] == "completed"
    assert result[0]["answer"] == "Quantum computing uses quantum mechanical phenomena."
    assert "created_at" in result[0]


def test_save_run_stores_mode_field(db, sample_state):
    result = save_run(
        db, sample_state, conversation_id="conv-abc", turn_number=1, mode="report"
    )
    assert result.mode == "report"


def test_get_all_reports_returns_empty_list_when_no_data(db):
    from focused_research_agent.database.repository import get_all_reports

    result = get_all_reports(db)
    assert result == []


def test_get_all_reports_returns_only_report_mode_runs(db, sample_state):
    from focused_research_agent.database.repository import get_all_reports

    save_run(
        db, sample_state, conversation_id="conv-chat", turn_number=1, mode="research"
    )
    save_run(
        db, sample_state, conversation_id="conv-report", turn_number=1, mode="report"
    )
    result = get_all_reports(db)
    assert len(result) == 1
    assert result[0]["conversation_id"] == "conv-report"


def test_get_all_reports_returns_correct_fields(db, sample_state):
    from focused_research_agent.database.repository import get_all_reports

    save_run(
        db, sample_state, conversation_id="conv-report", turn_number=1, mode="report"
    )
    result = get_all_reports(db)
    assert result[0]["conversation_id"] == "conv-report"
    assert result[0]["title"] == "What is quantum computing?"
    assert "created_at" in result[0]


def test_save_run_stores_and_retrieves_images(db, sample_state):
    """
    Verify that images are serialized on save and deserialized
    correctly when turns are retrieved.
    """
    save_run(db, sample_state, conversation_id="conv-img", turn_number=1)

    turns = get_conversation_turns(db, "conv-img")

    assert turns[0]["images"] == [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
    ]


def test_save_run_handles_none_images(db, sample_state):
    """
    Verify that None images is handled gracefully — stored as
    null and returned as None.
    """
    sample_state["images"] = None
    save_run(db, sample_state, conversation_id="conv-no-img", turn_number=1)

    turns = get_conversation_turns(db, "conv-no-img")

    assert turns[0]["images"] is None


def test_get_all_conversations_excludes_report_runs(db, sample_state):
    save_run(
        db, sample_state, conversation_id="conv-chat", turn_number=1, mode="research"
    )
    save_run(
        db, sample_state, conversation_id="conv-report", turn_number=1, mode="report"
    )

    result = get_all_conversations(db)

    assert len(result) == 1
    assert result[0]["conversation_id"] == "conv-chat"
