import json
import logging
import os
import random

from dotenv import load_dotenv
from groq import Groq
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

TEST_SET_SIZE   = 50    # ~50-100 range
OUTPUT_PATH     = "evaluation/test_set.json"
SEED            = 42

KNOWN_CATEGORIES = ["earphone", "phone", "speaker", "smartwatch", "laptop"]
KNOWN_ASPECTS    = ["battery", "sound", "build", "value", "mic", "display", "connectivity", "performance"]

SYSTEM_PROMPT = """You are creating a test set for evaluating a product-review RAG system.

Given ONE real customer review, generate:
1. A natural question a shopper would type into a search bar that this review
   would help answer. It must be answerable using ONLY the information in
   this review — do not invent facts the review doesn't support.
2. A ground-truth answer — a concise 1-2 sentence answer to that question,
   based strictly on what the review says.

The question should sound like a real user query (e.g. "does the boat
earphone have good battery life", "best earphone for calls"), not a
rephrasing of the review itself.

Respond with ONLY JSON:
{"question": "<question>", "ground_truth": "<answer>"}
"""


def sample_reviews_stratified(n: int) -> list[dict]:
    """
    Pull a stratified sample of reviews across category x aspect combos,
    so the test set covers the full space your router/retriever need to
    handle — not just whatever happens to be most common in the data.
    """
    logger.info("Sampling reviews stratified by category x aspect ...")

    query = text("""
        SELECT
            r.review_id, r.review_text, r.review_summary, r.rating,
            r.sentiment, r.primary_aspect, p.product_name, p.category, p.brand
        FROM reviews r
        JOIN products p ON r.product_id = p.product_id
        WHERE r.primary_aspect IS NOT NULL
          AND r.primary_aspect != 'other'
          AND r.review_text IS NOT NULL
          AND LENGTH(r.review_text) > 40
    """)

    with engine.connect() as conn:
        rows = [dict(row._mapping) for row in conn.execute(query).fetchall()]

    logger.info(f"✓ Pulled {len(rows):,} eligible reviews from PostgreSQL")

    # Group by (category, aspect)
    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["category"], r["primary_aspect"])
        buckets.setdefault(key, []).append(r)

    random.seed(SEED)
    per_bucket = max(1, n // max(len(buckets), 1))

    sampled = []
    for key, group in buckets.items():
        random.shuffle(group)
        sampled.extend(group[:per_bucket])

    random.shuffle(sampled)
    sampled = sampled[:n]

    logger.info(f"✓ Stratified sample across {len(buckets)} category/aspect buckets -> {len(sampled)} reviews")
    return sampled


def generate_qa_pair(review: dict) -> dict | None:
    """Ask Groq to generate a (question, ground_truth) pair grounded in one review."""
    review_text = review["review_text"] or review["review_summary"] or ""
    if not review_text.strip():
        return None

    user_content = (
        f"Product: {review['product_name']} ({review['category']}, brand: {review['brand']})\n"
        f"Rating: {review['rating']} | Sentiment: {review['sentiment']} | Aspect: {review['primary_aspect']}\n"
        f"Review: {review_text}"
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.5,
            max_tokens=250,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        question = parsed.get("question", "").strip()
        ground_truth = parsed.get("ground_truth", "").strip()

        if not question or not ground_truth:
            return None

        return {
            "question": question,
            "ground_truth": ground_truth,
            "source_review_id": review["review_id"],
            "category": review["category"],
            "aspect": review["primary_aspect"],
            "product_name": review["product_name"],
        }

    except Exception as e:
        logger.warning(f"QA generation failed for review {review['review_id']}: {e}")
        return None


def build_test_set(n: int = TEST_SET_SIZE) -> list[dict]:
    reviews = sample_reviews_stratified(n)

    test_set = []
    for i, review in enumerate(reviews, 1):
        pair = generate_qa_pair(review)
        if pair:
            test_set.append(pair)
        if i % 10 == 0:
            logger.info(f"Generated {len(test_set)}/{i} QA pairs so far ...")

    logger.info(f"✓ Final test set: {len(test_set)} QA pairs (requested {n})")
    return test_set


def save_test_set(test_set: list[dict], path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(test_set, f, indent=2)
    logger.info(f"✓ Saved test set to {path}")


if __name__ == "__main__":
    test_set = build_test_set(TEST_SET_SIZE)
    save_test_set(test_set)

    # Quick coverage summary
    from collections import Counter
    cat_counts = Counter(t["category"] for t in test_set)
    aspect_counts = Counter(t["aspect"] for t in test_set)
    logger.info(f"Category coverage: {dict(cat_counts)}")
    logger.info(f"Aspect coverage:   {dict(aspect_counts)}")
