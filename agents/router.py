import json
import logging
import os
from typing import Literal

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

RouteType = Literal["sql", "vector", "hybrid"]

SYSTEM_PROMPT = """You are a query router for a product-review RAG system.

Classify the user's query into exactly ONE of these routes:

- "sql": Query asks for structured aggregation/facts — averages, counts, price
  comparisons, filters on brand/category/price/rating with no opinion needed.
  Examples: "average rating of Boat earphones", "cheapest phone under 15000",
  "how many reviews does product X have"

- "vector": Query asks for subjective/opinion-based information found only in
  review text — sentiment, quality of an aspect, "best for X" without a
  strict numeric filter.
  Examples: "best earphone for mic quality", "do people like the battery life",
  "which speaker has bass complaints"

- "hybrid": Query needs BOTH structured filtering AND opinion/semantic
  retrieval — e.g. asks for "best X under price Y" or "top rated product for
  aspect Z".
  Examples: "best earphone for battery life under 2000",
  "top rated bluetooth speaker for bass"

Respond with ONLY a JSON object, no other text:
{"route": "sql" | "vector" | "hybrid", "reasoning": "<one short sentence>"}
"""


def classify_query(query: str) -> dict:
    """
    Classify a user query into one of: sql, vector, hybrid.

    Returns:
        {"route": "sql"|"vector"|"hybrid", "reasoning": str}
        Falls back to "hybrid" on any parsing/API failure (safest default —
        hybrid always has a chance of answering, worst case with noise).
    """
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)

        route = parsed.get("route", "hybrid")
        if route not in ("sql", "vector", "hybrid"):
            logger.warning(f"Router returned invalid route '{route}', defaulting to hybrid")
            route = "hybrid"

        logger.info(f"Route='{route}' | reasoning='{parsed.get('reasoning', '')}' | query='{query}'")

        return {"route": route, "reasoning": parsed.get("reasoning", "")}

    except Exception as e:
        logger.error(f"Router classification failed: {e} — defaulting to hybrid")
        return {"route": "hybrid", "reasoning": f"fallback due to error: {e}"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_queries = [
        "average rating of Boat earphones",
        "best earphone for mic quality",
        "best earphone for battery life under 2000",
    ]
    for q in test_queries:
        print(q, "->", classify_query(q))