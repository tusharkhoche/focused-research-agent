"""
SQLAlchemy engine and session management for the Focused Research
Agent.

This module creates the database engine from configuration, provides
a session factory for creating database sessions, and exposes a
get_db dependency function for use with FastAPI's dependency
injection system.

Architecturally, this module belongs to the database layer. It knows
about SQLAlchemy and database configuration only. It does not import
from the application layer, graph layer, or API layer.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from focused_research_agent.config.database_config import get_database_settings
from focused_research_agent.database.models import Base

engine = create_engine(
    get_database_settings().database_url,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """
    Create all database tables if they do not already exist.

    Safe to call multiple times — SQLAlchemy checks whether each table
    exists before creating it. Called once at application startup via
    the FastAPI lifespan or startup event.

    Returns:
        None
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session and guarantee it is closed after use.

    Used as a FastAPI dependency via Depends(get_db). Yields one
    Session for the duration of a request and closes it in a finally
    block so it is always released, even if an error occurs.

    Yields:
        Session: An active SQLAlchemy database session.
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
