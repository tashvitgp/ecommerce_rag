import functools
import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

from pipeline.retriever import hybrid_search
from pipeline.query_rewriter import multi_query_retrieve
from pipeline.reranker import rerank

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

KNOWN_CATEGORIES = ["earphone", "phone", "speaker", "smartwatch", "laptop"]
KNOWN_ASPECTS    = ["battery", "sound", "build", "value", "mic", "display", "connectivity", "performance"]


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


def run_vector_agent(
    query: str,
    top_k: int = 5,
    use_query_expansion: bool = True,
    use_reranking: bool = True,
    retrieve_pool_size: int = 15,
) -> dict:
    """
    Full vector agent flow:
        extract intent -> (multi-query expand ->) hybrid search (dense+sparse RRF)
        -> (cross-encoder rerank ->) top_k

    use_query_expansion: if True, runs the query + Groq-generated paraphrases
        through hybrid_search and merges by max RRF score (better recall).
    use_reranking: if True, cross-encoder reranks the retrieved pool before
        truncating to top_k (better precision).
    retrieve_pool_size: how many candidates to pull BEFORE reranking —
        should be >= top_k, larger gives reranker more to work with.

    Returns:
        {
            "query": str,
            "filter_used": {"category": ..., "aspect": ...},
            "results": [{"review_summary", "review_text", "product_name", ...}]
        }
    """
    intent = extract_intent(query)

    retrieve_fn = functools.partial(hybrid_search, category=intent["category"], aspect=intent["aspect"])

    if use_query_expansion:
        candidates = multi_query_retrieve(query, retrieve_fn, top_k=retrieve_pool_size)
    else:
        candidates = retrieve_fn(query, retrieve_pool_size)

    if use_reranking:
        final = rerank(query, candidates, top_k=top_k)
    else:
        final = candidates[:top_k]

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
            "rrf_score": round(r.get("rrf_score", 0.0), 4),
            "rerank_score": round(r["rerank_score"], 4) if "rerank_score" in r else None,
        }
        for r in final
    ]

    return {"query": query, "filter_used": intent, "results": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_vector_agent("best earphone for mic quality"), indent=2))
