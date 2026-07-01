"""
database/schemas.py
====================
SQLAlchemy ORM models for the Agentic RAG E-Commerce system.

Tables
------
  products              – dimension table, one row per unique product
  reviews               – fact table, one row per review (raw + NLP-enriched)
  conversational_cache  – query-response cache to avoid redundant LLM calls
  query_logs            – audit + eval log for every query processed by the system

CSV column → DB column mapping (flipkart_reviews.csv)
------------------------------------------------------
  product_name   → products.product_name
  product_price  → products.price
  Rate           → reviews.rating
  Review         → reviews.review_text
  Summary        → reviews.review_summary
  Sentiment      → reviews.sentiment
"""

import uuid

from sqlalchemy import (
    Column,
    String,
    Text,
    Numeric,
    Integer,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from database.connection import Base


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _uuid() -> str:
    """Generate a new UUID4 string. Used as column default."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------

class Product(Base):
    """
    Dimension table – one row per unique product.

    brand and category are nullable on ingest because they do not exist in the
    raw CSV. They are populated later by a post-ingest enrichment step that
    extracts the brand from product_name via NLP/regex and optionally
    classifies the category.
    """

    __tablename__ = "products"

    product_id   = Column(
        String(36),
        primary_key=True,
        default=_uuid,          # auto-generates UUID on every insert
        index=True,
    )
    product_name = Column(Text, nullable=False)

    # Enriched after ingest — nullable until step2_enrich_aspects.py runs
    brand        = Column(String(255), index=True, nullable=True)
    category     = Column(String(255), index=True, nullable=True)

    # Maps directly from CSV column "product_price"
    price        = Column(Numeric(10, 2), index=True, nullable=True)

    # Audit timestamps
    created_at   = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at   = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationship — cascade ensures reviews are deleted with their product
    reviews = relationship(
        "Review",
        back_populates="product",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Product id={self.product_id!r} name={self.product_name!r}>"


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

class Review(Base):
    """
    Fact table – one row per review.

    Raw CSV fields are stored as-is on ingest (step1_initialize_db.py).
    NLP-enriched fields (cleaned_text, primary_aspect, aspect_confidence) are
    populated by step2_enrich_aspects.py and start as NULL.
    """

    __tablename__ = "reviews"

    review_id   = Column(
        String(36),
        primary_key=True,
        default=_uuid,
        index=True,
    )
    product_id  = Column(
        String(36),
        ForeignKey("products.product_id", ondelete="CASCADE"),
        nullable=False,
        index=True,             # heavily queried — always index FK columns
    )

    # ------------------------------------------------------------------ #
    # Raw CSV fields                                                       #
    # ------------------------------------------------------------------ #

    # CSV: "Review" — full review body
    review_text    = Column(Text, nullable=True)

    # CSV: "Summary" — short headline written by the reviewer
    # Useful for UI citation cards and aspect extraction signal
    review_summary = Column(Text, nullable=True)

    # CSV: "Rate" — stored as Numeric to safely handle both "4" and "4.2"
    rating         = Column(Numeric(3, 1), nullable=True)

    # CSV: "Sentiment" — pre-computed label ("Positive" / "Negative" / "Neutral")
    # Store the CSV value directly; avoid recomputing what is already given
    sentiment      = Column(String(50), nullable=True)

    # Original review date from the CSV if present; nullable if CSV lacks it
    review_date    = Column(DateTime, nullable=True)

    # ------------------------------------------------------------------ #
    # NLP-enriched fields  (step2_enrich_aspects.py populates these)      #
    # ------------------------------------------------------------------ #

    cleaned_text      = Column(Text, nullable=True)
    primary_aspect    = Column(
        String(255),
        index=True,
        nullable=True,          # "battery" | "sound" | "build" | "value" | "mic" | "display"
    )
    aspect_confidence = Column(Numeric(5, 4), nullable=True)  # 0.0000 – 1.0000

    # Audit timestamp — when this row was inserted into the DB
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationship back to parent product
    product = relationship("Product", back_populates="reviews")

    # ------------------------------------------------------------------ #
    # Composite indexes for the most common SQL agent query patterns       #
    # ------------------------------------------------------------------ #

    __table_args__ = (
        # "Give me all battery-related reviews for product X"
        Index("ix_reviews_product_aspect", "product_id", "primary_aspect"),
        # "Give me all 5-star reviews for product X"
        Index("ix_reviews_product_rating", "product_id", "rating"),
        # "Give me recent positive reviews for product X"
        Index("ix_reviews_product_sentiment", "product_id", "sentiment"),
    )

    def __repr__(self) -> str:
        return (
            f"<Review id={self.review_id!r} "
            f"product={self.product_id!r} "
            f"rating={self.rating!r} "
            f"aspect={self.primary_aspect!r}>"
        )


# ---------------------------------------------------------------------------
# ConversationalCache
# ---------------------------------------------------------------------------

class ConversationalCache(Base):
    """
    Query-response cache.

    Stores the SHA-256 hash of a query as the primary key so identical queries
    can be served from cache without hitting the LLM again.

    hit_count  – incremented on every cache hit; useful for analytics
    expires_at – optional TTL so stale answers are not served indefinitely;
                 set to None for permanent cache entries
    """

    __tablename__ = "conversational_cache"

    query_hash      = Column(String(64), primary_key=True, index=True)  # SHA-256 hex digest
    original_query  = Column(Text, nullable=False)
    cached_response = Column(Text, nullable=False)

    # Cache management
    hit_count  = Column(Integer, default=0, nullable=False)
    expires_at = Column(DateTime, nullable=True)   # NULL = never expires

    # Audit
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<ConversationalCache hash={self.query_hash!r} "
            f"hits={self.hit_count!r}>"
        )


# ---------------------------------------------------------------------------
# QueryLog
# ---------------------------------------------------------------------------

class QueryLog(Base):
    """
    Audit and evaluation log – one row per query processed by the pipeline.

    Every query is written here along with:
      • the rewritten / expanded query (after query_rewriter.py)
      • which agent route was chosen by router.py ("sql" | "vector" | "hybrid")
      • the IDs of chunks retrieved from Qdrant (stored as a JSON string)
      • RAGAS evaluation scores computed by evaluate_pipeline.py

    This table is the foundation of your evaluation analytics notebook
    (notebooks/03_eval_results.ipynb). It lets you answer:
      – Which query types have the lowest faithfulness?
      – Did reranking improve answer relevancy over time?
      – How does the hybrid route compare to pure vector?
    """

    __tablename__ = "query_logs"

    log_id = Column(
        String(36),
        primary_key=True,
        default=_uuid,
        index=True,
    )

    # ------------------------------------------------------------------ #
    # Query fields                                                         #
    # ------------------------------------------------------------------ #

    query_text      = Column(Text, nullable=False)
    rewritten_query = Column(Text, nullable=True)   # after query_rewriter.py expands it

    # Router decision: "sql" | "vector" | "hybrid"
    agent_route     = Column(String(50), nullable=True, index=True)

    # JSON array of Qdrant point IDs returned by the retriever, e.g.:
    # '["abc-123", "def-456"]'
    retrieved_chunk_ids = Column(Text, nullable=True)

    # ------------------------------------------------------------------ #
    # RAGAS evaluation scores (populated by evaluate_pipeline.py)         #
    # ------------------------------------------------------------------ #

    faithfulness_score  = Column(Numeric(5, 4), nullable=True)  # 0.0 – 1.0
    answer_relevancy    = Column(Numeric(5, 4), nullable=True)  # 0.0 – 1.0
    context_precision   = Column(Numeric(5, 4), nullable=True)  # 0.0 – 1.0
    context_recall      = Column(Numeric(5, 4), nullable=True)  # 0.0 – 1.0

    # Custom metrics (defined in evaluation/evaluate_pipeline.py)
    aspect_hit_rate     = Column(Numeric(5, 4), nullable=True)  # did retrieval use the right aspect?
    contradiction_score = Column(Numeric(5, 4), nullable=True)  # did answer surface conflicting opinions?

    # Audit
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<QueryLog id={self.log_id!r} "
            f"route={self.agent_route!r} "
            f"faithfulness={self.faithfulness_score!r}>"
        )
