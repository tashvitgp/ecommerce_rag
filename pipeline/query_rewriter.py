import json
import logging
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You rewrite product-review search queries into multiple paraphrased
versions to improve retrieval recall.

Generate 3 alternative phrasings of the user's query that preserve the exact
same intent (same product category, same aspect, same constraints like price)
but vary the wording — synonyms, reordering, more/less specific phrasing.

Do NOT introduce new constraints or drop existing ones (e.g. don't add or
remove a price filter that wasn't in the original).

Respond with ONLY a JSON object:
{"variants": ["<query 1>", "<query 2>", "<query 3>"]}
"""


def expand_query(query: str, n: int = 3) -> list[str]:
    """
    Multi-query expansion: generate n paraphrased variants of the query via
    Groq, to be run through hybrid_search independently and merged/deduped.

    Returns the original query plus up to n variants. Falls back to just
    the original query on any failure.
    """
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        variants = parsed.get("variants", [])[:n]
        logger.info(f"Expanded '{query}' -> {variants}")
        return [query] + variants

    except Exception as e:
        logger.error(f"Query expansion failed: {e} — using original query only")
        return [query]


def multi_query_retrieve(query: str, retrieve_fn, top_k: int = 10, n_variants: int = 3) -> list[dict]:
    queries = expand_query(query, n=n_variants)

    merged: dict[str, dict] = {}
    for q in queries:
        # Pass top_k explicitly as a keyword argument
        results = retrieve_fn(q, top_k=top_k)
        for r in results:
            pid = r["id"]
            if pid not in merged or r["rrf_score"] > merged[pid]["rrf_score"]:
                merged[pid] = r

    ranked = sorted(merged.values(), key=lambda r: r["rrf_score"], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import functools
    from pipeline.retriever import hybrid_search, qdrant # Import qdrant instance here

    try:
        query = "best earphone for battery life"
        retrieve_fn = functools.partial(hybrid_search, category="earphone", aspect="battery")
        results = multi_query_retrieve(query, retrieve_fn, top_k=5)
        for r in results:
            print(round(r["rrf_score"], 4), r["payload"].get("review_summary", "")[:60])
    finally:
        qdrant.close() # Clean cleanup here too