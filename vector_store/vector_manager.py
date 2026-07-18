
import argparse
import json
import logging
import os
import re
import time
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



DATABASE_URL    = os.getenv("DATABASE_URL")
QDRANT_PATH     = os.getenv("QDRANT_PATH", "qdrant_storage")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "product_reviews")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384      
BATCH_SIZE      = 256       
BM25_VOCAB_PATH = os.getenv("BM25_VOCAB_PATH", "bm25_vocab.json")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in .env file")


engine = create_engine(DATABASE_URL)

logger.info(f"Connecting to local Qdrant at path: {QDRANT_PATH}")
qdrant = QdrantClient(path=QDRANT_PATH)

# ---------------------------------------------------------------------------
# Embedding model — lazy loaded
# ---------------------------------------------------------------------------

_model: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    """Load embedding model once and cache it."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"✓ Model loaded — output dim: {EMBEDDING_DIM}")
    return _model


# ---------------------------------------------------------------------------
# BM25 sparse vector helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def build_bm25_vectors(
    texts: list[str],
    vocab: dict[str, int],
    idf: dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    """
    Compute BM25 sparse vectors for a list of texts.

    Returns list of dicts with 'indices' and 'values' keys,
    matching Qdrant's SparseVector format.

    BM25 parameters:
        k1 = 1.5  — term frequency saturation (higher = more weight on freq)
        b  = 0.75 — length normalisation (1.0 = full normalisation)
    """
    # Average document length for normalisation
    lengths   = [len(tokenize(t)) for t in texts]
    avg_dl    = np.mean(lengths) if lengths else 1.0

    results = []
    for text, dl in zip(texts, lengths):
        tokens     = tokenize(text)
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        indices = []
        values  = []
        for token, freq in tf.items():
            if token not in vocab:
                continue
            # BM25 formula
            tf_norm  = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avg_dl))
            bm25_val = tf_norm * idf.get(token, 0.0)
            if bm25_val > 0:
                indices.append(vocab[token])
                values.append(float(bm25_val))

        results.append({"indices": indices, "values": values})

    return results


def build_corpus_vocab_idf(texts: list[str]) -> tuple[dict[str, int], dict[str, float]]:
    """
    Build vocabulary (token → index) and IDF scores from the full corpus.
    Called once before indexing — O(n) over all review texts.
    """
    logger.info("Building BM25 vocabulary and IDF from corpus ...")
    doc_freq: dict[str, int] = {}
    n_docs = len(texts)

    for text in texts:
        tokens = set(tokenize(text))   
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    vocab = {
        token: idx
        for idx, (token, freq) in enumerate(doc_freq.items())
        if freq >= 2
    }

    idf = {
        token: float(np.log((n_docs - freq + 0.5) / (freq + 0.5) + 1))
        for token, freq in doc_freq.items()
        if token in vocab
    }

    logger.info(f"✓ Vocabulary: {len(vocab):,} tokens from {n_docs:,} documents")
    return vocab, idf


def save_bm25_vocab_idf(vocab: dict[str, int], idf: dict[str, float]) -> None:
    """
    Persist the vocab/idf mapping to disk so query-time code (vector_agent.py)
    can build sparse vectors using the EXACT same token->index mapping that
    was used to build the sparse vectors stored in Qdrant.

    Without this file, vector_agent.py would build its own vocab from
    scratch at query time, and the sparse vector indices would not line up
    with what's stored in the collection — search would silently return
    wrong/garbage results.
    """
    with open(BM25_VOCAB_PATH, "w") as f:
        json.dump({"vocab": vocab, "idf": idf}, f)
    logger.info(f"✓ Saved BM25 vocab/idf to {BM25_VOCAB_PATH}")


def setup_collection(vocab_size: int) -> None:
    """
    Create Qdrant collection with hybrid vector config.

    Creates two vector spaces:
        'dense'  — 384-dim float vectors for semantic search
        'sparse' — BM25 sparse vectors for keyword search

    Safe to call multiple times — skips if collection already exists.
    """
    existing = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME in existing:
        count = qdrant.count(COLLECTION_NAME).count
        logger.info(
            f"Collection '{COLLECTION_NAME}' already exists "
            f"with {count:,} points. Skipping creation."
        )
        return

    logger.info(f"Creating collection '{COLLECTION_NAME}' ...")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,  # cosine similarity for semantic search
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )

    logger.info(f"✓ Collection '{COLLECTION_NAME}' created")


def load_reviews_from_postgres() -> list[dict]:
    """
    Fetch enriched reviews from PostgreSQL.

    Only loads reviews where:
        primary_aspect IS NOT NULL  — enriched by step2
        primary_aspect != 'other'   — has meaningful aspect signal

    Joins with products to get product_name and category for payload.
    """
    logger.info("Loading enriched reviews from PostgreSQL ...")

    query = text("""
        SELECT
            r.review_id,
            r.product_id,
            p.product_name,
            p.category,
            p.brand,
            r.review_text,
            r.review_summary,
            r.cleaned_text,
            r.primary_aspect,
            r.aspect_confidence,
            r.rating,
            r.sentiment
        FROM reviews r
        JOIN products p ON r.product_id = p.product_id
        WHERE r.primary_aspect IS NOT NULL
          AND r.primary_aspect != 'other'
          AND r.cleaned_text IS NOT NULL
          AND r.cleaned_text != ''
        ORDER BY r.review_id
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    reviews = [dict(row._mapping) for row in rows]
    logger.info(f"✓ Loaded {len(reviews):,} enriched reviews from PostgreSQL")
    return reviews



def deduplicate_reviews(reviews: list[dict]) -> list[dict]:
    """
    Drop duplicate reviews that appear across color/size/variant SKUs of the
    same underlying product listing (common in scraped e-commerce data —
    the exact same review text gets attached to every variant's product_id).

    Dedup key: cleaned_text (falls back to review_text). Keeps the FIRST
    occurrence encountered (query is ordered by review_id, so this is stable).

    This runs before embedding/indexing so duplicate points never enter
    Qdrant — cleaner retrieval results and no wasted embedding compute.
    """
    seen: set[str] = set()
    deduped = []
    dropped = 0

    for r in reviews:
        key = (r.get("cleaned_text") or r.get("review_text") or "").strip().lower()
        if not key:
            deduped.append(r)  # keep rows with no text, nothing to dedupe on
            continue
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(r)

    if dropped:
        logger.info(
            f"✓ Deduplicated reviews: dropped {dropped:,} duplicate(s) "
            f"across product variants — {len(deduped):,} unique reviews remain "
            f"(was {len(reviews):,})"
        )

    return deduped


def build_index() -> None:


    reviews = load_reviews_from_postgres()
    if not reviews:
        logger.error("No enriched reviews found. Run step2_enrich_aspects2.py first.")
        return

    reviews = deduplicate_reviews(reviews)

    total = len(reviews)

    all_texts = [r["cleaned_text"] or r["review_text"] or "" for r in reviews]
    vocab, idf = build_corpus_vocab_idf(all_texts)

    # Persist vocab/idf so vector_agent.py can build matching query-time
    # sparse vectors. Always saved, even if the collection already exists,
    # in case this is the first time you're wiring up vector_agent.py.
    save_bm25_vocab_idf(vocab, idf)

    setup_collection(vocab_size=len(vocab))

    existing_count = qdrant.count(COLLECTION_NAME).count
    if existing_count >= total:
        logger.info(
            f"Collection already has {existing_count:,} points "
            f"(expected {total:,}). Nothing to index."
        )
        return


    model = get_model()


    logger.info(f"Indexing {total:,} reviews in batches of {BATCH_SIZE} ...")
    t_start     = time.time()
    processed   = 0
    n_batches   = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, start in enumerate(range(0, total, BATCH_SIZE), 1):
        batch   = reviews[start : start + BATCH_SIZE]
        texts   = [r["cleaned_text"] or r["review_text"] or "" for r in batch]

        dense_vecs = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,  # normalise for cosine similarity
        )


        sparse_vecs = build_bm25_vectors(texts, vocab, idf)


        points = []
        for i, review in enumerate(batch):
            points.append(
                models.PointStruct(
                    id=processed + i,        
                    vector={
                        "dense":  dense_vecs[i].tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vecs[i]["indices"],
                            values=sparse_vecs[i]["values"],
                        ),
                    },
                    payload={

                        "review_id":        str(review["review_id"]),
                        "product_id":       str(review["product_id"]),

                        "product_name":     str(review["product_name"] or ""),
                        "category":         str(review["category"] or ""),
                        "brand":            str(review["brand"] or ""),

                        "primary_aspect":   str(review["primary_aspect"] or ""),
                        "aspect_confidence": float(review["aspect_confidence"] or 0),

                        "review_text":      str(review["review_text"] or ""),
                        "review_summary":   str(review["review_summary"] or ""),

                        # Quality signals — used for SQL pre-filtering
                        "rating":           float(review["rating"] or 0),
                        "sentiment":        str(review["sentiment"] or ""),
                    },
                )
            )

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        processed += len(batch)
        elapsed    = time.time() - t_start
        rate       = processed / elapsed if elapsed > 0 else 1
        eta        = (total - processed) / rate

        logger.info(
            f"Batch {batch_idx}/{n_batches} | "
            f"{processed:,}/{total:,} ({100*processed/total:.1f}%) | "
            f"{rate:.0f} rev/s | "
            f"ETA: {eta:.0f}s"
        )

    elapsed = time.time() - t_start
    final_count = qdrant.count(COLLECTION_NAME).count

    logger.info("=" * 55)
    logger.info("✓ Indexing complete")
    logger.info(f"  Indexed    : {final_count:,} points")
    logger.info(f"  Time       : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Speed      : {processed/elapsed:.0f} reviews/sec")
    logger.info("=" * 55)
    logger.info("Next step: run agents/router.py")



def test_search() -> None:
    """
    Run a few test queries to verify the index is working correctly.
    Tests both dense-only and hybrid search.
    """
    model = get_model()

    test_queries = [
        ("best battery life earphone",      "earphone", "battery"),
        ("good sound quality speaker",       "speaker",  "sound"),
        ("phone display brightness outdoor", "phone",    "display"),
        ("smartwatch connectivity bluetooth","smartwatch","connectivity"),
        ("laptop performance gaming speed",  "laptop",   "performance"),
    ]

    logger.info("\n" + "=" * 55)
    logger.info("RUNNING TEST SEARCHES")
    logger.info("=" * 55)

    for query, expected_category, expected_aspect in test_queries:
        logger.info(f"\nQuery: '{query}'")
        logger.info(f"Expected: category={expected_category}, aspect={expected_aspect}")

        # Encode query
        query_vec = model.encode(query, normalize_embeddings=True).tolist()

        # Dense-only search
        dense_results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=("dense", query_vec),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=expected_category),
                    ),
                    models.FieldCondition(
                        key="primary_aspect",
                        match=models.MatchValue(value=expected_aspect),
                    ),
                ]
            ),
            limit=3,
            with_payload=True,
        )

        logger.info(f"  Top {len(dense_results)} results (dense + aspect filter):")
        for rank, hit in enumerate(dense_results, 1):
            logger.info(
                f"    {rank}. [{hit.payload['primary_aspect']}] "
                f"score={hit.score:.3f} | "
                f"{hit.payload['product_name'][:40]} | "
                f"{hit.payload['review_summary'][:60]}"
            )

    logger.info("\n✓ Test search complete")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qdrant vector index.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test searches after indexing (or standalone if already indexed).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and rebuild from scratch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.reset:
        existing = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION_NAME in existing:
            qdrant.delete_collection(COLLECTION_NAME)
            logger.info(f"✓ Deleted collection '{COLLECTION_NAME}'")

    build_index()

    if args.test:
        test_search()
