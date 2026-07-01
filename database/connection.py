import logging
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

# Attempt to load .env using UTF-8, fall back to UTF-16 if file has a BOM/UTF-16 encoding
try:
    load_dotenv(encoding='utf-8')
except UnicodeDecodeError:
    # Common when .env was saved as UTF-16; try that encoding as a fallback
    load_dotenv(encoding='utf-16')

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is not set.\n"
        "Create a .env file in the project root with:\n"
        "  DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/ecommerce_rag"
    )


engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    pass



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
