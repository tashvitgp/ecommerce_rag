"""
database/connection.py
=======================
Centralised SQLAlchemy engine, session factory, and Base class.

All other modules import from here — never create engines elsewhere.

Environment variables (set in .env)
------------------------------------
    DATABASE_URL  postgresql://user:password@host:5432/dbname
"""

import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load environment variables from .env (local dev only)
# In production (Docker / cloud), these come from the environment directly.
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Database URL — never hardcode credentials here
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is not set.\n"
        "Create a .env file in the project root with:\n"
        "  DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/ecommerce_rag"
    )

# ---------------------------------------------------------------------------
# Engine
#
# pool_size      — number of persistent connections kept alive in the pool
# max_overflow   — extra connections allowed during traffic bursts (beyond pool_size)
# pool_pre_ping  — run "SELECT 1" before handing out a connection to detect
#                  stale/dropped connections; prevents cryptic errors mid-request
# ---------------------------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# ---------------------------------------------------------------------------
# Session factory
#
# autocommit=False  — transactions must be committed explicitly (safe default)
# autoflush=False   — don't auto-flush before every query (gives more control)
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ---------------------------------------------------------------------------
# Declarative Base
#
# Using SQLAlchemy 2.0 class-based style.
# All models in schemas.py inherit from this Base.
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Startup connection check
#
# Call this once when the app / pipeline starts to fail fast if the DB
# is unreachable, rather than getting a cryptic error mid-query.
# ---------------------------------------------------------------------------

def verify_connection() -> None:
    """
    Run a lightweight SELECT 1 to confirm the DB is reachable.
    Raises on failure so the caller can decide to exit or retry.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✓ Database connection verified")
    except Exception as exc:
        logger.error(f"✗ Cannot connect to database: {exc}")
        raise


# ---------------------------------------------------------------------------
# FastAPI dependency
#
# Use this as a dependency in your route functions:
#
#   from database.connection import get_db
#   from sqlalchemy.orm import Session
#
#   @app.get("/products")
#   def list_products(db: Session = Depends(get_db)):
#       ...
# ---------------------------------------------------------------------------

def get_db():
    """
    Yield a database session and ensure it is closed after the request,
    even if an exception is raised.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
