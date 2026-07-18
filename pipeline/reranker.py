import logging
from typing import Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL} ...")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("✓ Reranker loaded")
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 5, text_field: str = "review_text") -> list[dict]:
    """
    Rerank a list of retrieved candidates against the query using a
    cross-encoder. Expects each candidate to be a dict with a 'payload' key
    (matching retriever.py's hybrid_search output format).

    Returns candidates sorted by cross-encoder score, truncated to top_k,
    with a 'rerank_score' field added to each.
    """
    if not candidates:
        return []

    model = get_reranker()

    pairs = []
    for c in candidates:
        text = c["payload"].get(text_field) or c["payload"].get("review_summary", "") or ""
        pairs.append([query, text])

    scores = model.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.retriever import hybrid_search

    query = "best earphone for battery life"
    candidates = hybrid_search(query, category="earphone", aspect="battery", top_k=10)
    top = rerank(query, candidates, top_k=5)
    for r in top:
        print(round(r["rerank_score"], 4), r["payload"].get("review_summary", "")[:60])