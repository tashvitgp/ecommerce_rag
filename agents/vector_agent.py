import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from qdrant_client import models

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL      = "llama-3.1-8b-instant"
QDRANT_PATH     = os.getenv("QDRANT_PATH", "qdrant_storage")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "product_reviews")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VOCAB_PATH      = os.getenv("BM25_VOCAB_PATH", "bm25_vocab.json")  # see note below

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
qdrant = QdrantClient(path=QDRANT_PATH)

KNOWN_CATEGORIES = ["earphone", "phone", "speaker", "smartwatch", "laptop"]
KNOWN_ASPECTS    = ["battery", "sound", "build", "value", "mic", "display", "connectivity", "performance"]

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
    Load the vocab/idf saved during indexing (see vector_manager.py addition below).
    Required so the query-time sparse vector uses the same token->index mapping
    as the vectors stored in Qdrant.
    """
    global _vocab, _idf
    if _vocab is None or _idf is None:
        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(
                f"{VOCAB_PATH} not found. Add the vocab/idf save step to "
                f"vector_manager.py's build_index() and rerun indexing."
            )
        with open(VOCAB_PATH, "r") as f:
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
    """Build a BM25-style sparse vector for the query using the saved vocab/idf."""
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


def extract_intent(query: str) -> dict:
    """
    Use Groq to extract category + aspect from the query for pre-filtering.
    Falls back to no filter if the LLM can't confidently match a known value.
    """
    system_prompt = f"""Extract the product category and review aspect from the query.

Known categories: {KNOWN_CATEGORIES}
Known aspects: {KNOWN_ASPECTS}

Respond with ONLY JSON: {{"category": "<one of known categories or null>", "aspect": "<one of known aspects or null>"}}
Use null if the query doesn't clearly match any known value — do not guess.
"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        category = parsed.get("category")
        aspect = parsed.get("aspect")
        return {
            "category": category if category in KNOWN_CATEGORIES else None,
            "aspect": aspect if aspect in KNOWN_ASPECTS else None,
        }
    except Exception as e:
        logger.error(f"Intent extraction failed: {e} — proceeding with no filter")
        return {"category": None, "aspect": None}


def build_filter(category: Optional[str], aspect: Optional[str]) -> Optional[models.Filter]:
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


def run_vector_agent(query: str, top_k: int = 5) -> dict:
    """
    Full vector agent flow: extract intent -> filter -> hybrid search -> RRF -> top_k.

    Returns:
        {
            "query": str,
            "filter_used": {"category": ..., "aspect": ...},
            "results": [{"review_summary", "review_text", "product_name", "rating", ...}]
        }
    """
    intent = extract_intent(query)
    query_filter = build_filter(intent["category"], intent["aspect"])

    embed_model = get_embed_model()
    dense_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    sparse_vec = build_query_sparse_vector(query)

# Modern clean approach for named vector search
# 1. Dense Vector Query
    dense_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vec,               # Raw list of floats/embeddings
        using="dense",                 # Targets the "dense" vector configuration
        query_filter=query_filter,
        limit=top_k * 3,
        with_payload=True,
    ).points

    # 2. Sparse Vector Query
    sparse_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        # CHANGED: Use the raw sparse vector structure directly
        query=models.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
        using="sparse",                # CHANGED: Specifies the name of your sparse vector field
        query_filter=query_filter,
        limit=top_k * 3,
        with_payload=True,
    ).points

    fused = reciprocal_rank_fusion(dense_results, sparse_results)[:top_k]

    results = [
        {
            "review_id": r["payload"].get("review_id"),
            "product_name": r["payload"].get("product_name"),
            "brand": r["payload"].get("brand"),
            "category": r["payload"].get("category"),
            "primary_aspect": r["payload"].get("primary_aspect"),
            "rating": r["payload"].get("rating"),
            "sentiment": r["payload"].get("sentiment"),
            "review_summary": r["payload"].get("review_summary"),
            "review_text": r["payload"].get("review_text"),
            "rrf_score": round(r["rrf_score"], 4),
        }
        for r in fused
    ]

    return {"query": query, "filter_used": intent, "results": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_vector_agent("best earphone for mic quality"), indent=2))