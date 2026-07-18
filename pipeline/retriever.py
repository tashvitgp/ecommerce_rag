import logging
import os
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)

QDRANT_PATH     = os.getenv("QDRANT_PATH", "qdrant_storage")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "product_reviews")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BM25_VOCAB_PATH = os.getenv("BM25_VOCAB_PATH", "bm25_vocab.json")

qdrant = QdrantClient(path=QDRANT_PATH)

_embed_model: Optional[SentenceTransformer] = None
_vocab: Optional[dict] = None
_idf: Optional[dict] = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def get_bm25_lookup() -> tuple[dict, dict]:
    """
    Load the vocab/idf saved by vector_store/vector_manager.py at indexing
    time. Required so query-time sparse vectors use the same token->index
    mapping as the vectors stored in Qdrant.
    """
    global _vocab, _idf
    if _vocab is None or _idf is None:
        import json
        if not os.path.exists(BM25_VOCAB_PATH):
            raise FileNotFoundError(
                f"{BM25_VOCAB_PATH} not found. Run "
                f"`python -m vector_store.vector_manager` first to generate it."
            )
        with open(BM25_VOCAB_PATH, "r") as f:
            data = json.load(f)
        _vocab, _idf = data["vocab"], data["idf"]
    return _vocab, _idf


def tokenize(text: str) -> list[str]:
    import re
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def build_query_sparse_vector(query: str) -> models.SparseVector:
    """Build a BM25-style sparse vector for a query using the saved vocab/idf."""
    vocab, idf = get_bm25_lookup()
    tokens = tokenize(query)
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    indices, values = [], []
    for token, freq in tf.items():
        if token in vocab:
            indices.append(vocab[token])
            values.append(float(freq) * idf.get(token, 0.0))

    return models.SparseVector(indices=indices, values=values)


def build_filter(category: Optional[str] = None, aspect: Optional[str] = None) -> Optional[models.Filter]:
    conditions = []
    if category:
        conditions.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
    if aspect:
        conditions.append(models.FieldCondition(key="primary_aspect", match=models.MatchValue(value=aspect)))
    return models.Filter(must=conditions) if conditions else None


def reciprocal_rank_fusion(dense_results, sparse_results, k: int = 60) -> list[dict]:
    """
    Combine dense + sparse ranked lists using RRF:
        score = sum(1 / (k + rank)) across lists a point appears in.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, hit in enumerate(dense_results, start=1):
        scores[hit.id] = scores.get(hit.id, 0) + 1 / (k + rank)
        payloads[hit.id] = hit.payload

    for rank, hit in enumerate(sparse_results, start=1):
        scores[hit.id] = scores.get(hit.id, 0) + 1 / (k + rank)
        payloads.setdefault(hit.id, hit.payload)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": pid, "rrf_score": score, "payload": payloads[pid]} for pid, score in ranked]


def hybrid_search(
    query: str,
    category: Optional[str] = None,
    aspect: Optional[str] = None,
    top_k: int = 10,
) -> list[dict]:
    query_filter = build_filter(category, aspect)

    embed_model = get_embed_model()
    dense_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    sparse_vec = build_query_sparse_vector(query)

    # For dense search
    dense_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vec,          # Pass the vector directly here
        using="dense",            # Specify the name of the vector configuration
        query_filter=query_filter,
        limit=top_k * 3,
        with_payload=True,
    ).points                      # Extract the list of points from the response object

    # For sparse search
    sparse_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_vec,         # Pass the sparse vector model directly here
        using="sparse",           # Specify the name of the sparse vector configuration
        query_filter=query_filter,
        limit=top_k * 3,
        with_payload=True,
    ).points                      # Extract the list of points from the response object

    fused = reciprocal_rank_fusion(dense_results, sparse_results)[:top_k]
    return fused

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = hybrid_search("best earphone for battery life", category="earphone", aspect="battery", top_k=5)
    for r in results:
        print(round(r["rrf_score"], 4), r["payload"].get("review_summary", "")[:60])