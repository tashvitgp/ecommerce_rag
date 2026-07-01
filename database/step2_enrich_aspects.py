"""
database/step2_enrich_aspects.py
==================================
NLP enrichment pipeline — reads every review from PostgreSQL, tags it with a
primary aspect label, and writes the result back.

What this script adds to each review row
-----------------------------------------
    cleaned_text       normalised review text (lowercased, stripped, de-noised)
    primary_aspect     dominant aspect: battery | sound | build | value |
                                        mic | display | performance | other
    aspect_confidence  model confidence score (0.0 – 1.0)

How aspect classification works
--------------------------------
We use zero-shot classification (facebook/bart-large-mnli) which means:
  - No training data needed
  - Just provide candidate labels and it scores each one
  - Pick the highest scoring label as primary_aspect

Why zero-shot and not fine-tuned?
-----------------------------------
For a portfolio project this is the right call:
  - Works out of the box on any domain
  - Explainable in interviews ("I used BART MNLI for zero-shot classification
    because I didn't have labelled aspect data")
  - Accuracy is good enough (70–80%) for downstream retrieval filtering
  - step2 can be re-run later with a fine-tuned model to show improvement

Performance
-----------
205K reviews is large. This script:
  - Processes in batches of 64 to avoid OOM
  - Skips already-enriched rows (safe to re-run / resume after crash)
  - Logs progress every 1000 rows
  - Estimated time: 45–90 min on CPU | 10–15 min on GPU

Usage
-----
    python -m database.step2_enrich_aspects

    # To process only a sample first (recommended before full run):
    python -m database.step2_enrich_aspects --sample 5000
"""

import argparse
import logging
import re
import time
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from database.connection import SessionLocal, verify_connection

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aspect labels
# These are the candidate labels passed to the zero-shot classifier.
# Keep them short and non-overlapping — the model scores each one.
# ---------------------------------------------------------------------------

ASPECT_LABELS = [
    "battery life",
    "sound quality",
    "build quality",
    "value for money",
    "microphone quality",
    "display quality",
    "performance and speed",
    "comfort and fit",
    "connectivity",
    "other",
]

# Map verbose label → short DB label stored in primary_aspect column
LABEL_MAP = {
    "battery life":         "battery",
    "sound quality":        "sound",
    "build quality":        "build",
    "value for money":      "value",
    "microphone quality":   "mic",
    "display quality":      "display",
    "performance and speed":"performance",
    "comfort and fit":      "comfort",
    "connectivity":         "connectivity",
    "other":                "other",
}

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise raw review text for NLP processing.

    Steps:
      1. Lowercase
      2. Remove special characters (keep letters, digits, punctuation)
      3. Collapse multiple whitespace into single space
      4. Strip leading/trailing whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\-]", " ", text)  # keep basic punctuation
    text = re.sub(r"\s+", " ", text)                      # collapse whitespace
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Lazy model loader — only loads when first called
# ---------------------------------------------------------------------------

_classifier = None

def get_classifier():
    """
    Load the zero-shot classification pipeline once and cache it.
    Uses facebook/bart-large-mnli — best accuracy for zero-shot text classification.

    Lazy loading means the model isn't downloaded until the first batch
    is processed, which keeps startup fast.
    """
    global _classifier
    if _classifier is None:
        logger.info("Loading zero-shot classification model ...")
        logger.info("(First run will download ~1.6GB — subsequent runs use cache)")
        from transformers import pipeline
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,      # -1 = CPU; change to 0 for GPU if available
        )
        logger.info("✓ Model loaded")
    return _classifier


# ---------------------------------------------------------------------------
# Aspect classification
# ---------------------------------------------------------------------------

def classify_aspects(texts: list[str]) -> list[tuple[str, float]]:
    """
    Run zero-shot classification on a batch of texts.

    Returns a list of (primary_aspect_label, confidence_score) tuples,
    one per input text.

    Falls back to ("other", 0.0) for empty or unclassifiable texts.
    """
    classifier = get_classifier()
    results = []

    for text in texts:
        if not text or len(text.strip()) < 5:
            results.append(("other", 0.0))
            continue
        try:
            output = classifier(
                text[:512],          # truncate — BART has 1024 token limit, safe at 512 chars
                candidate_labels=ASPECT_LABELS,
                multi_label=False,   # pick ONE primary aspect only
            )
            top_label      = output["labels"][0]
            top_score      = round(float(output["scores"][0]), 4)
            short_label    = LABEL_MAP.get(top_label, "other")
            results.append((short_label, top_score))
        except Exception as e:
            logger.warning(f"Classification failed for text snippet: {text[:50]!r} — {e}")
            results.append(("other", 0.0))

    return results


# ---------------------------------------------------------------------------
# Batch writer
# ---------------------------------------------------------------------------

def write_batch(db: Session, updates: list[dict]) -> None:
    """
    Bulk-update a batch of review rows with enriched fields.

    Uses raw SQL for speed — SQLAlchemy ORM update() on 205K rows
    would be very slow row-by-row.
    """
    db.execute(
        text("""
            UPDATE reviews
            SET
                cleaned_text      = :cleaned_text,
                primary_aspect    = :primary_aspect,
                aspect_confidence = :aspect_confidence
            WHERE review_id = :review_id
        """),
        updates,
    )
    db.commit()


# ---------------------------------------------------------------------------
# Main enrichment loop
# ---------------------------------------------------------------------------

def enrich(sample: Optional[int] = None, batch_size: int = 64) -> None:
    """
    Main enrichment pipeline.

    Args:
        sample:     If set, only process this many reviews (for testing).
        batch_size: Number of reviews to classify in one model call.
                    Larger = faster but more RAM. 64 is safe for 8GB RAM.
    """
    verify_connection()
    db: Session = SessionLocal()

    try:
        # ---------------------------------------------------------------- #
        # Count how many reviews still need enrichment
        # Skips already-processed rows — makes script safe to re-run
        # ---------------------------------------------------------------- #
        total_pending = db.execute(
            text("SELECT COUNT(*) FROM reviews WHERE primary_aspect IS NULL")
        ).scalar()

        if total_pending == 0:
            logger.info("✓ All reviews already enriched. Nothing to do.")
            return

        logger.info(f"Reviews pending enrichment: {total_pending:,}")

        if sample:
            logger.info(f"Sample mode: processing {sample:,} reviews only")
            limit_clause = f"LIMIT {sample}"
        else:
            limit_clause = ""

        # ---------------------------------------------------------------- #
        # Fetch only unenriched rows
        # ---------------------------------------------------------------- #
        rows = db.execute(
            text(f"""
                SELECT review_id, review_text, review_summary
                FROM reviews
                WHERE primary_aspect IS NULL
                ORDER BY review_id
                {limit_clause}
            """)
        ).fetchall()

        logger.info(f"Fetched {len(rows):,} reviews to process")

        # ---------------------------------------------------------------- #
        # Process in batches
        # ---------------------------------------------------------------- #
        total       = len(rows)
        processed   = 0
        t_start     = time.time()

        for batch_start in range(0, total, batch_size):
            batch = rows[batch_start : batch_start + batch_size]

            # Prefer review_summary for classification (shorter, more focused)
            # Fall back to review_text if summary is empty
            texts = [
                (row.review_summary or row.review_text or "")
                for row in batch
            ]

            # Clean text
            cleaned = [clean_text(t) for t in texts]

            # Classify aspects
            aspect_results = classify_aspects(cleaned)

            # Build update payload
            updates = [
                {
                    "review_id":        batch[i].review_id,
                    "cleaned_text":     cleaned[i],
                    "primary_aspect":   aspect_results[i][0],
                    "aspect_confidence": aspect_results[i][1],
                }
                for i in range(len(batch))
            ]

            # Write to DB
            write_batch(db, updates)

            processed += len(batch)

            # ------------------------------------------------------------ #
            # Progress logging every 1000 rows
            # ------------------------------------------------------------ #
            if processed % 1000 < batch_size or processed == total:
                elapsed      = time.time() - t_start
                rate         = processed / elapsed if elapsed > 0 else 0
                remaining    = (total - processed) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {processed:,}/{total:,} reviews "
                    f"({100 * processed / total:.1f}%) | "
                    f"{rate:.0f} reviews/sec | "
                    f"ETA: {remaining/60:.1f} min"
                )

        # ---------------------------------------------------------------- #
        # Final summary
        # ---------------------------------------------------------------- #
        elapsed = time.time() - t_start
        logger.info("=" * 55)
        logger.info("✓ Enrichment complete")
        logger.info(f"  Processed : {processed:,} reviews")
        logger.info(f"  Total time: {elapsed/60:.1f} minutes")
        logger.info(f"  Avg speed : {processed/elapsed:.0f} reviews/sec")
        logger.info("=" * 55)

        # ---------------------------------------------------------------- #
        # Aspect distribution — useful sanity check
        # ---------------------------------------------------------------- #
        logger.info("Aspect distribution in enriched reviews:")
        distribution = db.execute(
            text("""
                SELECT primary_aspect, COUNT(*) as count
                FROM reviews
                WHERE primary_aspect IS NOT NULL
                GROUP BY primary_aspect
                ORDER BY count DESC
            """)
        ).fetchall()

        for row in distribution:
            bar = "█" * (row.count // max(1, processed // 40))
            logger.info(f"  {row.primary_aspect:<15} {row.count:>8,}  {bar}")

        logger.info("Next step: run vector_store/vector_manager.py")

    finally:
        db.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich reviews with NLP aspect tags."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process N reviews (for testing before full run). E.g. --sample 1000",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Reviews per classification batch (default: 64). Reduce if OOM.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enrich(sample=args.sample, batch_size=args.batch_size)