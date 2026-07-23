# import json
# import logging
# import os
# from typing import Literal

# from dotenv import load_dotenv
# from groq import Groq

# load_dotenv()

# logger = logging.getLogger(__name__)

# GROQ_MODEL = "llama-3.1-8b-instant"
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# RouteType = Literal["sql", "vector", "hybrid"]

# SYSTEM_PROMPT = """You are a query router for a product-review RAG system.

# Classify the user's query into exactly ONE of these routes:

# - "sql": Query asks for structured aggregation/facts — averages, counts, price
#   comparisons, filters on brand/category/price/rating with no opinion needed.
#   Examples: "average rating of Boat earphones", "cheapest phone under 15000",
#   "how many reviews does product X have"

# - "vector": Query asks for subjective/opinion-based information found only in
#   review text — sentiment, quality of an aspect, "best for X" without a
#   strict numeric filter.
#   Examples: "best earphone for mic quality", "do people like the battery life",
#   "which speaker has bass complaints"

# - "hybrid": Query needs BOTH structured filtering AND opinion/semantic
#   retrieval — e.g. asks for "best X under price Y" or "top rated product for
#   aspect Z".
#   Examples: "best earphone for battery life under 2000",
#   "top rated bluetooth speaker for bass"

# Respond with ONLY a JSON object, no other text:
# {"route": "sql" | "vector" | "hybrid", "reasoning": "<one short sentence>"}
# """


# def classify_query(query: str) -> dict:
#     """
#     Classify a user query into one of: sql, vector, hybrid.

#     Returns:
#         {"route": "sql"|"vector"|"hybrid", "reasoning": str}
#         Falls back to "hybrid" on any parsing/API failure (safest default —
#         hybrid always has a chance of answering, worst case with noise).
#     """
#     try:
#         response = client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": query},
#             ],
#             temperature=0,
#             max_tokens=150,
#             response_format={"type": "json_object"},
#         )

#         raw = response.choices[0].message.content
#         parsed = json.loads(raw)

#         route = parsed.get("route", "hybrid")
#         if route not in ("sql", "vector", "hybrid"):
#             logger.warning(f"Router returned invalid route '{route}', defaulting to hybrid")
#             route = "hybrid"

#         logger.info(f"Route='{route}' | reasoning='{parsed.get('reasoning', '')}' | query='{query}'")

#         return {"route": route, "reasoning": parsed.get("reasoning", "")}

#     except Exception as e:
#         logger.error(f"Router classification failed: {e} — defaulting to hybrid")
#         return {"route": "hybrid", "reasoning": f"fallback due to error: {e}"}


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     test_queries = [
#         "average rating of Boat earphones",
#         "best earphone for mic quality",
#         "best earphone for battery life under 2000",
#     ]
#     for q in test_queries:
#         print(q, "->", classify_query(q))



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

SYSTEM_PROMPT = """You are a query router for a product-review RAG system with two data sources:

1. SQL DATABASE (structured columns only):
   products: product_name, brand, category, price
   reviews:  rating, sentiment, primary_aspect (battery/sound/build/value/mic/display/connectivity/performance)
   SQL can ONLY answer questions using these exact columns — aggregates (average/count/min/max),
   filtering by brand/category/price/rating, or counting reviews by sentiment/aspect label.
   SQL CANNOT answer anything that requires reading the actual text of a review — it has no
   columns for specific features, specs, compatibility, or any fact only mentioned in free text.

2. VECTOR STORE (semantic search over review TEXT):
   Use this for ANY question whose answer would only appear inside what a reviewer actually
   wrote — even if the question sounds like a simple yes/no factual question.

THE KEY TEST: ask yourself "could this be answered by a SQL column, or do I need to read what
someone actually wrote in a review?" If the fact (a spec, a feature, a compatibility claim, a
comfort/quality/ease-of-use claim, anything product-specific) only exists as free text INSIDE
a review, it is a VECTOR question — even if phrased as "does X have Y" or "is X true".

Classify into exactly ONE route:

- "sql": Only when the question is answerable purely from structured columns above.
  Examples: "average rating of Boat earphones", "cheapest phone under 15000",
  "how many reviews does product X have", "count of positive reviews for brand Y"

- "vector": Question needs information that only exists in review TEXT — specs, features,
  compatibility, comfort, ease of setup, "does it have X", "is it good for Y", opinions.
  Examples: "does the MSI GF63 have a numeric keypad", "does this laptop have a 144Hz display",
  "is this headset compatible with most devices", "how long does the battery last",
  "best earphone for mic quality", "is this smartwatch worth buying"

- "hybrid": Needs BOTH a structured filter AND text-based opinion/fact retrieval together —
  e.g. price/brand/rating filter combined with a feature or opinion question.
  Examples: "best earphone for battery life under 2000", "top rated bluetooth speaker for bass"

IMPORTANT: A question phrased as a factual yes/no ("does it have...", "is it...", "how long...")
is USUALLY "vector", not "sql" — unless the fact is literally one of the structured columns
listed above (price, rating, brand, category, sentiment label, aspect label, or a count/average
of these). Do not default to "sql" just because a question sounds factual.

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
        "does the MSI GF63 have a numeric keypad",
        "does this laptop have a 144Hz display",
        "is this headset compatible with most devices",
        "how long does the boAt Rockerz 510 battery last",
        "best earphone for battery life under 2000",
    ]
    for q in test_queries:
        print(q, "->", classify_query(q))