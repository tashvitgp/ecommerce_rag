# import pandas as pd
# import uuid
# from database.connection import engine
# from database.schemas import Base

# df = pd.read_csv("data/flipkart_reviews.csv")

# # normalize/clean columns based on actual CSV headers
# df['Review'] = df['Review'].fillna('')
# df['Summary'] = df['Summary'].fillna('')
# df['review_text'] = df['Summary'] + ". " + df['Review']

# # CSV uses lowercase 'product_name' and 'product_price'
# unique_products = df['product_name'].unique()
# product_id_mapping = {name: str(uuid.uuid4()) for name in unique_products}
# df['product_id'] = df['product_name'].map(product_id_mapping)

# df['review_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

# df['brand'] = df['product_name'].apply(lambda x: str(x).split()[0] if pd.notnull(x) else "Unknown")
# # keep a consistent column name for downstream code
# df['product_name'] = df['product_name']
# df['price'] = pd.to_numeric(df['product_price'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
# df['rating'] = pd.to_numeric(df['Rate'], errors='coerce')
# df['category'] = "General"
# df['timestamp'] = pd.Timestamp.now()

# products = df[['product_id', 'product_name', 'brand', 'price', 'category']].drop_duplicates(subset=['product_id'])
# reviews_raw = df[['review_id', 'product_id', 'review_text', 'rating', 'timestamp']]

# Base.metadata.create_all(bind=engine)

# products.to_sql("products", engine, if_exists="append", index=False, method="multi", chunksize=2000)
# reviews_raw.to_sql("reviews", engine, if_exists="append", index=False, method="multi", chunksize=2000)




import logging
import uuid
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from database.connection import engine
from database.schemas import Base



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_PATH = Path("data/flipkart_reviews.csv")

# Known brand names for smarter extraction
# Extend this list as you explore your dataset
KNOWN_BRANDS = [
    "boAt", "Samsung", "OnePlus", "Realme", "Redmi", "Xiaomi",
    "Apple", "Sony", "JBL", "Noise", "Fastrack", "Zebronics",
    "Philips", "Lenovo", "HP", "Dell", "Asus", "Acer",
]

# Keyword-based category inference
# Key = category label stored in DB, Value = substrings to match in product_name
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "earphone": [
        "earphone", "earbud", "headphone", "airdopes",
        "headset", "tws", "neckband", "bassheads",
    ],
    "phone": [
        "mobile", "phone", "smartphone", "redmi",
        "realme", "galaxy", "iphone", "oneplus",
    ],
    "laptop": ["laptop", "notebook", "chromebook", "vivobook", "ideapad"],
    "smartwatch": ["smartwatch", "watch", "band", "fitness tracker"],
    "speaker": ["speaker", "soundbar", "bluetooth speaker"],
    "tablet": ["tablet", "ipad", "tab "],
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_brand(product_name: str) -> str | None:
    """
    Extract brand from product name using a known-brands lookup first,
    falling back to the first word of the name.

    Returns None if product_name is null/empty.
    """
    if not product_name or pd.isnull(product_name):
        return None
    name_lower = str(product_name).lower()
    for brand in KNOWN_BRANDS:
        if brand.lower() in name_lower:
            return brand
    # Fallback: use first word (still better than nothing)
    return str(product_name).split()[0]


def infer_category(product_name: str) -> str | None:
    """
    Infer a broad category from product name keywords.

    Returns None if no keyword matches — honest NULL beats wrong "General".
    """
    if not product_name or pd.isnull(product_name):
        return None
    name_lower = str(product_name).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return None  # unknown — enrichment step can improve this later


def clean_price(price_series: pd.Series) -> pd.Series:
    """Strip currency symbols and commas, then coerce to float."""
    return pd.to_numeric(
        price_series.astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )


def normalise_sentiment(sentiment_series: pd.Series) -> pd.Series:
    """
    Normalise raw Sentiment values to: 'Positive' | 'Negative' | 'Neutral'
    Handles casing variants and unknown values gracefully.
    """
    mapping = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral":  "Neutral",
    }
    return (
        sentiment_series
        .fillna("Neutral")
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna("Neutral")      # anything unrecognised → Neutral
    )


# ---------------------------------------------------------------------------
# Idempotency guard
# ---------------------------------------------------------------------------

def already_ingested() -> bool:
    """Return True if the products table already has rows."""
    try:
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
            return (count or 0) > 0
    except Exception:
        # Table may not exist yet — not ingested
        return False


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def load_and_clean_csv() -> pd.DataFrame:
    """Load CSV and return a cleaned DataFrame with all required columns."""
    logger.info(f"Loading CSV from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded {len(df):,} rows | columns: {list(df.columns)}")

    # ------------------------------------------------------------------ #
    # Validate required columns exist
    # ------------------------------------------------------------------ #
    required = {"product_name", "product_price", "Rate", "Review", "Summary", "Sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # ------------------------------------------------------------------ #
    # Clean text columns
    # ------------------------------------------------------------------ #
    df["Review"]  = df["Review"].fillna("").str.strip()
    df["Summary"] = df["Summary"].fillna("").str.strip()

    # review_summary = raw CSV Summary (preserved as its own column)
    df["review_summary"] = df["Summary"]

    # review_text = Summary + ". " + Review (enriched combined field for embedding)
    df["review_text"] = df.apply(
        lambda r: (r["Summary"] + ". " + r["Review"]).strip(". ").strip()
        if r["Summary"] else r["Review"],
        axis=1,
    )

    # ------------------------------------------------------------------ #
    # Numeric columns
    # ------------------------------------------------------------------ #
    df["price"]  = clean_price(df["product_price"])
    df["rating"] = pd.to_numeric(df["Rate"], errors="coerce")

    # ------------------------------------------------------------------ #
    # Sentiment
    # ------------------------------------------------------------------ #
    df["sentiment"] = normalise_sentiment(df["Sentiment"])

    # ------------------------------------------------------------------ #
    # Brand + category inference
    # ------------------------------------------------------------------ #
    df["brand"]    = df["product_name"].apply(extract_brand)
    df["category"] = df["product_name"].apply(infer_category)

    # ------------------------------------------------------------------ #
    # IDs
    # ------------------------------------------------------------------ #
    unique_products        = df["product_name"].unique()
    product_id_map         = {name: str(uuid.uuid4()) for name in unique_products}
    df["product_id"]       = df["product_name"].map(product_id_map)
    df["review_id"]        = [str(uuid.uuid4()) for _ in range(len(df))]

    # No date column in Flipkart CSV — store NULL cleanly
    df["review_date"] = None

    logger.info(
        f"Cleaned: {df['product_id'].nunique():,} unique products | "
        f"{len(df):,} reviews | "
        f"brand coverage: {df['brand'].notna().sum():,} | "
        f"category coverage: {df['category'].notna().sum():,}"
    )
    return df


def build_products_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and deduplicate the products dimension DataFrame."""
    products_df = (
        df[["product_id", "product_name", "brand", "price", "category"]]
        .drop_duplicates(subset=["product_id"])
        .reset_index(drop=True)
    )
    logger.info(f"Products to insert: {len(products_df):,}")
    return products_df


def build_reviews_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the reviews fact DataFrame — all CSV-sourced columns included."""
    reviews_df = df[[
        "review_id",
        "product_id",
        "review_text",
        "review_summary",   # CSV: Summary
        "rating",           # CSV: Rate
        "sentiment",        # CSV: Sentiment (normalised)
        "review_date",      # NULL — no date in CSV
    ]].reset_index(drop=True)
    logger.info(f"Reviews to insert: {len(reviews_df):,}")
    return reviews_df


def insert_df(df: pd.DataFrame, table: str, label: str) -> None:
    """
    Insert a DataFrame into the given table in 2000-row chunks.
    Raises on failure so the caller can handle it.
    """
    logger.info(f"Inserting {len(df):,} {label} ...")
    t0 = time.time()
    try:
        df.to_sql(
            table,
            engine,
            if_exists="append",     # table already created by create_all()
            index=False,
            method="multi",         # batches values into single INSERT statements
            chunksize=2000,
        )
        elapsed = time.time() - t0
        logger.info(f"✓ {label} inserted in {elapsed:.1f}s")
    except Exception as exc:
        logger.error(f"✗ Failed to insert {label}: {exc}")
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Idempotency guard — safe to re-run without duplicating data
    # ------------------------------------------------------------------ #
    if already_ingested():
        logger.warning(
            "Products table already has rows. "
            "Skipping ingest to avoid duplicates. "
            "Drop the tables and re-run if you need a fresh load."
        )
        return

    # ------------------------------------------------------------------ #
    # 2. Create all tables (CREATE TABLE IF NOT EXISTS)
    # ------------------------------------------------------------------ #
    logger.info("Creating database tables ...")
    Base.metadata.create_all(bind=engine)
    logger.info("✓ Tables ready")

    # ------------------------------------------------------------------ #
    # 3. Load + clean CSV
    # ------------------------------------------------------------------ #
    df = load_and_clean_csv()

    # ------------------------------------------------------------------ #
    # 4. Build dimension and fact DataFrames
    # ------------------------------------------------------------------ #
    products_df = build_products_df(df)
    reviews_df  = build_reviews_df(df)

    # ------------------------------------------------------------------ #
    # 5. Insert — products first (FK parent before child)
    # ------------------------------------------------------------------ #
    insert_df(products_df, "products", "products")
    insert_df(reviews_df,  "reviews",  "reviews")

    # ------------------------------------------------------------------ #
    # 6. Final summary
    # ------------------------------------------------------------------ #
    logger.info("=" * 50)
    logger.info("✓ Ingest complete")
    logger.info(f"  Products : {len(products_df):,}")
    logger.info(f"  Reviews  : {len(reviews_df):,}")
    logger.info("Next step  : run database/step2_enrich_aspects.py")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()