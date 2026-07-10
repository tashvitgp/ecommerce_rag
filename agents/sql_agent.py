import logging
import os
import re

from dotenv import load_dotenv
from groq import Groq
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# ---------------------------------------------------------------------------
# Schema description handed to the LLM — mirrors schemas.py exactly
# ---------------------------------------------------------------------------

SCHEMA_DESCRIPTION = """
Table: products
  product_id   VARCHAR(36) PRIMARY KEY
  product_name TEXT
  brand        VARCHAR(255)   -- nullable
  category     VARCHAR(255)   -- nullable, e.g. 'earphone', 'phone', 'speaker'
  price        NUMERIC(10,2)  -- nullable

Table: reviews
  review_id         VARCHAR(36) PRIMARY KEY
  product_id        VARCHAR(36) REFERENCES products(product_id)
  review_text       TEXT
  review_summary    TEXT
  rating            NUMERIC(3,1)   -- 1.0 to 5.0
  sentiment         VARCHAR(50)    -- 'Positive' | 'Negative' | 'Neutral'
  review_date       TIMESTAMP
  cleaned_text      TEXT
  primary_aspect    VARCHAR(255)   -- 'battery' | 'sound' | 'build' | 'value' | 'mic' | 'display'
  aspect_confidence NUMERIC(5,4)

Relationship: reviews.product_id -> products.product_id (many reviews per product)
"""

SYSTEM_PROMPT = f"""You are a PostgreSQL query generator for a product-review database.

Schema:
{SCHEMA_DESCRIPTION}

Rules:
- Generate ONLY a single SELECT statement. Never DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE.
- Always JOIN products and reviews when the query needs fields from both.
- Use aggregate functions (AVG, COUNT, MIN, MAX) for aggregation questions.
- Always add a reasonable LIMIT (default 20) unless the query is a pure aggregate (COUNT/AVG returning one row).
- Use ILIKE for case-insensitive brand/category text matching.
- Respond with ONLY the raw SQL statement. No markdown, no explanation, no semicolon-wrapped code fences.
"""

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|REVOKE|CREATE)\b",
    re.IGNORECASE,
)


def generate_sql(query: str) -> str:
    """Ask Groq to generate a SELECT statement for the given natural language query."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=400,
    )
    sql = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    sql = re.sub(r"^```sql\s*|```$", "", sql, flags=re.IGNORECASE).strip()
    sql = sql.rstrip(";")
    return sql


def is_safe_select(sql: str) -> bool:
    """Guardrail: only allow single SELECT statements, no mutating keywords."""
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return False
    if FORBIDDEN_KEYWORDS.search(sql):
        return False
    if ";" in sql.rstrip(";"):  # blocks stacked statements
        return False
    return True


def run_sql_agent(query: str) -> dict:
    """
    Full SQL agent flow: generate SQL -> validate -> execute -> return rows.

    Returns:
        {
            "sql": str,
            "success": bool,
            "results": list[dict] | None,
            "error": str | None,
        }
    """
    sql = generate_sql(query)
    logger.info(f"Generated SQL: {sql}")

    if not is_safe_select(sql):
        logger.warning(f"Rejected unsafe SQL: {sql}")
        return {
            "sql": sql,
            "success": False,
            "results": None,
            "error": "Generated SQL failed safety validation (must be a single SELECT statement).",
        }

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
        results = [dict(row) for row in rows]
        return {"sql": sql, "success": True, "results": results, "error": None}

    except SQLAlchemyError as e:
        logger.error(f"SQL execution failed: {e}")
        return {"sql": sql, "success": False, "results": None, "error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run_sql_agent("average rating of Boat earphones"))