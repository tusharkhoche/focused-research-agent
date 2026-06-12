"""
Database repository for the Focused Research Agent.

This module is the only file in the project that reads from and writes
to the database. All other modules that need data access call these
functions — they never interact with SQLAlchemy sessions directly.

Architecturally, this module belongs to the database layer and
implements the Repository Pattern. It abstracts storage concerns away
from the application layer. Switching databases requires changing only
this file and database.py — no application or graph code changes.

List fields (queries, sources, citations, errors) are serialized to
JSON strings on save and deserialized back to Python lists on read.
This conversion is transparent to the rest of the application.
"""

import logging
import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from focused_research_agent.database.models import ConversationRun

logger = logging.getLogger(__name__)


def _serialize(value: list | None) -> str | None:
    """
    Serialize a Python list to a JSON string for database storage.

    Args:
        value: List to serialize, or None.

    Returns:
        str | None: JSON string if value is a list, None if value is
            None.
    """
    if value is None:
        return None
    return json.dumps(value)


def _deserialize(value: str | None) -> list | None:
    """
    Deserialize a JSON string from the database back to a Python list.

    Args:
        value: JSON string to deserialize, or None.

    Returns:
        list | None: Python list if value is a string, None if value
            is None.
    """
    if value is None:
        return None
    return json.loads(value)


def save_run(
    db: Session,
    state: dict,
    conversation_id: str,
    turn_number: int,
    mode: str = "research",
) -> ConversationRun:
    """
    Save a completed research run to the database.

    Creates a new ConversationRun row from the normalized research
    state. Sets conversation_title from the first 60 characters of
    the question on turn 1 only. Serializes list fields to JSON
    strings before storing.

    Args:
        db: Active SQLAlchemy database session.
        state: Normalized research result dict from the application
            layer.
        conversation_id: UUID string linking this run to its
            conversation.
        turn_number: Position of this run within the conversation
            (1-based).

    Returns:
        ConversationRun: The saved model instance with its database
            ID populated.
    """
    now = datetime.now(timezone.utc)

    conversation_title = None
    if turn_number == 1:
        conversation_title = state["question"][:60]

    run = ConversationRun(
        conversation_id=conversation_id,
        turn_number=turn_number,
        conversation_title=conversation_title,
        run_id=state["run_id"],
        question=state["question"],
        status=state["status"],
        scope=state.get("scope"),
        queries=_serialize(state.get("queries")),
        sources=_serialize(state.get("sources")),
        answer=state.get("answer"),
        citations=_serialize(state.get("citations")),
        errors=_serialize(state.get("errors")),
        images=_serialize(state.get("images")),
        created_at=now,
        updated_at=now,
        mode=mode,
    )

    db.add(run)
    db.commit()
    db.refresh(run)
    logger.info(
        "Run saved. conversation_id=%s turn=%d mode=%s run_id=%s",
        conversation_id,
        turn_number,
        mode,
        state["run_id"],
    )
    return run


def get_conversation_history(
    db: Session,
    conversation_id: str,
    max_turns: int,
) -> list[dict]:
    """
    Fetch the most recent turns of a conversation for context
    threading.

    Returns turns in chronological order (oldest first) so they can
    be included directly in the LLM prompt as conversation history.

    Args:
        db: Active SQLAlchemy database session.
        conversation_id: UUID string identifying the conversation.
        max_turns: Maximum number of recent turns to return. Older
            turns beyond this limit are excluded to manage LLM
            context size.

    Returns:
        list[dict]: List of turn dicts in chronological order, each
            containing turn, question, answer, and scope keys.
            Empty list if the conversation has no turns yet.
    """
    runs = (
        db.query(ConversationRun)
        .filter(ConversationRun.conversation_id == conversation_id)
        .order_by(ConversationRun.turn_number.desc())
        .limit(max_turns)
        .all()
    )

    runs = list(reversed(runs))

    history = []
    for run in runs:
        history.append(
            {
                "turn": run.turn_number,
                "question": run.question,
                "answer": run.answer,
                "scope": run.scope,
            }
        )

    logger.debug(
        "Conversation history fetched. conversation_id=%s turns=%d",
        conversation_id,
        len(history),
    )
    return history


def get_all_conversations(db: Session) -> list[dict]:
    """
    Fetch a summary list of all conversations for the history sidebar.

    Returns one entry per conversation using only the first turn of
    each, ordered newest first. Used by the GET /api/v1/conversations
    endpoint and the chat UI history panel.

    Args:
        db: Active SQLAlchemy database session.

    Returns:
        list[dict]: List of conversation summary dicts, each
            containing conversation_id, title, and created_at keys.
            Empty list if no conversations exist yet.
    """
    runs = (
        db.query(ConversationRun)
        .filter(ConversationRun.turn_number == 1, ConversationRun.mode != "report")
        .order_by(ConversationRun.created_at.desc())
        .all()
    )

    conversations = []
    for run in runs:
        conversations.append(
            {
                "conversation_id": run.conversation_id,
                "title": run.conversation_title,
                "created_at": run.created_at.isoformat(),
            }
        )

    logger.debug("All conversations fetched. count=%d", len(conversations))
    return conversations


def get_conversation_turns(
    db: Session,
    conversation_id: str,
) -> list[dict]:
    """
    Fetch all turns of a conversation in chronological order.

    Returns the complete research data for every turn, including
    deserialized list fields. Used by the GET conversations endpoint
    to load a full conversation into the chat UI.

    Args:
        db: Active SQLAlchemy database session.
        conversation_id: UUID string identifying the conversation.

    Returns:
        list[dict]: List of complete turn dicts in chronological
            order. Empty list if the conversation does not exist.
    """
    runs = (
        db.query(ConversationRun)
        .filter(ConversationRun.conversation_id == conversation_id)
        .order_by(ConversationRun.turn_number.asc())
        .all()
    )

    turns = []
    for run in runs:
        turns.append(
            {
                "turn_number": run.turn_number,
                "run_id": run.run_id,
                "question": run.question,
                "status": run.status,
                "scope": run.scope,
                "queries": _deserialize(run.queries),
                "sources": _deserialize(run.sources),
                "answer": run.answer,
                "citations": _deserialize(run.citations),
                "errors": _deserialize(run.errors),
                "created_at": run.created_at.isoformat(),
                "images": _deserialize(run.images),
            }
        )
    logger.debug(
        "Conversation turns fetched. conversation_id=%s count=%d",
        conversation_id,
        len(turns),
    )
    return turns


def get_all_reports(db: Session) -> list[dict]:
    """
    Fetch a summary list of all report runs for the report history
    sidebar.

    Returns one entry per report ordered newest first. Filters by
    mode='report' to exclude chat and research runs.

    Args:
        db: Active SQLAlchemy database session.

    Returns:
        list[dict]: List of report summary dicts containing
            conversation_id, title, and created_at keys.
            Empty list if no reports exist yet.
    """
    runs = (
        db.query(ConversationRun)
        .filter(ConversationRun.mode == "report")
        .order_by(ConversationRun.created_at.desc())
        .all()
    )

    reports = []
    for run in runs:
        reports.append(
            {
                "conversation_id": run.conversation_id,
                "title": run.conversation_title,
                "created_at": run.created_at.isoformat(),
            }
        )
    logger.debug("All reports fetched. count=%d", len(reports))
    return reports
