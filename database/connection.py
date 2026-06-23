import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:191102@localhost:5432/ecommerce_rag")

# Create the SQLAlchemy engine 
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class to handle database transactions safely
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class that our schema models will inherit from
Base = declarative_base()

# Dependency function to get a database session (useful for FastAPI later)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()