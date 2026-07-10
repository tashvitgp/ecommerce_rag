
import argparse
import logging
import re
import time
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.connection import SessionLocal, verify_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ASPECTS: dict[str, dict] = {
    "battery": {
        "keywords": ["battery", "charge", "charging", "backup", "drain", "mah",
                     "battery life", "fast charge", "battery drain", "dies fast",
                     "battery backup", "power"],
        "seeds": [
            "battery life is excellent and lasts all day",
            "battery drains very fast and needs frequent charging",
            "the battery backup is poor and disappoints",
            "charges quickly with fast charging support",
            "battery dies within a few hours of use",
        ],
    },
    "sound": {
        "keywords": ["sound", "audio", "bass", "treble", "volume", "music",
                     "speaker quality", "noise cancel", "acoustic", "clarity",
                     "loud", "distortion"],
        "seeds": [
            "sound quality is excellent with deep bass",
            "audio clarity is crisp and clear for music",
            "bass is heavy and treble is balanced",
            "volume is loud enough for outdoor use",
            "noise cancellation works perfectly in noisy environments",
        ],
    },
    "build": {
        "keywords": ["build quality", "build", "plastic", "sturdy", "flimsy",
                     "cheap material", "durable", "broke", "crack", "premium feel",
                     "metal body", "material", "finish", "fragile", "solid"],
        "seeds": [
            "build quality is solid and feels premium",
            "the plastic body feels cheap and flimsy",
            "very durable and did not break after dropping",
            "the material quality is excellent and sturdy",
            "cracked after light use which is disappointing",
        ],
    },
    "value": {
        "keywords": ["value for money", "worth it", "price", "expensive", "cheap",
                     "budget", "overpriced", "affordable", "cost", "best buy",
                     "waste of money", "not worth", "good deal", "pricing"],
        "seeds": [
            "excellent value for money at this price point",
            "very overpriced for the features it offers",
            "best budget option available in this range",
            "not worth the price at all very disappointing",
            "affordable and delivers good performance for the cost",
        ],
    },
    "mic": {
        "keywords": ["mic", "microphone", "call quality", "voice clarity",
                     "calls", "voice", "noise reduction", "caller", "speakerphone"],
        "seeds": [
            "microphone quality is crystal clear during calls",
            "call quality is poor and the other person cannot hear me",
            "voice clarity on calls is excellent even in noisy areas",
            "the mic picks up too much background noise",
        ],
    },
    "display": {
        "keywords": ["display", "screen", "resolution", "brightness", "touchscreen",
                     "panel", "amoled", "lcd", "hd", "full hd", "4k", "refresh rate",
                     "sunlight visibility", "color", "contrast"],
        "seeds": [
            "display quality is sharp and vibrant with good colors",
            "screen brightness is excellent even in direct sunlight",
            "the resolution is crisp and clear for watching videos",
            "touchscreen is very responsive and smooth",
        ],
    },
    "performance": {
        "keywords": ["performance", "lag", "speed", "fast", "slow", "processor",
                     "hang", "freeze", "smooth", "heating", "heat", "ram",
                     "multitask", "gaming"],
        "seeds": [
            "performance is smooth with no lag during multitasking",
            "the phone heats up very quickly under load",
            "processor is fast and handles gaming well",
            "freezes frequently and hangs during normal use",
        ],
    },
    "comfort": {
        "keywords": ["comfort", "comfortable", "fit", "lightweight", "heavy",
                     "pain", "ergonomic", "wearable", "strap", "ear tips", "neck", "wrist"],
        "seeds": [
            "very comfortable to wear for long hours",
            "the ear tips fit perfectly and do not fall out",
            "lightweight design makes it easy to carry all day",
            "causes ear pain after extended use",
        ],
    },
    "connectivity": {
        "keywords": ["bluetooth", "wifi", "connectivity", "pairing", "range",
                     "signal", "disconnect", "wireless", "connection", "stable",
                     "drops", "network", "5g", "4g"],
        "seeds": [
            "bluetooth connectivity is stable with no drops",
            "wifi signal is strong and maintains fast speeds",
            "keeps disconnecting from bluetooth which is frustrating",
            "pairing with devices is quick and seamless",
        ],
    },
    "other": {
        "keywords": [],
        "seeds": [
            "overall experience is good and I recommend this product",
            "product arrived on time and packaging was intact",
        ],
    },
}

ASPECT_NAMES = list(ASPECTS.keys())


def clean_text(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    raw = raw.lower()
    raw = re.sub(r"[^a-z0-9\s\.\,\!\?\-]", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def keyword_classify(text: str) -> Optional[tuple[str, float]]:
    if not text:
        return None
    matches: dict[str, int] = {}
    for aspect, config in ASPECTS.items():
        if aspect == "other":
            continue
        count = sum(1 for kw in config["keywords"] if kw in text)
        if count > 0:
            matches[aspect] = count
    if not matches:
        return None
    return (max(matches, key=matches.get), 0.75)


class TFIDFClassifier:
    def __init__(self):
        logger.info("Building TF-IDF classifier from seed sentences ...")
        self._aspect_labels: list[str] = []
        seeds: list[str] = []
        for aspect, config in ASPECTS.items():
            for seed in config["seeds"]:
                self._aspect_labels.append(aspect)
                seeds.append(seed)
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        self._seed_matrix = self._vectorizer.fit_transform(seeds)
        logger.info(f"✓ TF-IDF ready — {len(seeds)} seeds across {len(ASPECT_NAMES)} aspects")

    def classify(self, texts: list[str]) -> list[tuple[str, float]]:
        if not texts:
            return []
        matrix = self._vectorizer.transform(texts)
        sims   = cosine_similarity(matrix, self._seed_matrix)
        results = []
        for row in sims:
            best_idx   = int(np.argmax(row))
            best_score = float(row[best_idx])
            aspect     = self._aspect_labels[best_idx] if best_score >= 0.05 else "other"
            results.append((aspect, round(best_score, 4)))
        return results


_tfidf: Optional[TFIDFClassifier] = None

def get_tfidf() -> TFIDFClassifier:
    global _tfidf
    if _tfidf is None:
        _tfidf = TFIDFClassifier()
    return _tfidf


def classify_batch(texts: list[str]) -> list[tuple[str, float]]:
    results:         list[Optional[tuple[str, float]]] = []
    fallback_idx:    list[int] = []
    fallback_texts:  list[str] = []

    for i, text in enumerate(texts):
        if not text or len(text.strip()) < 5:
            results.append(("other", 0.0))
            continue
        kw = keyword_classify(text)
        if kw:
            results.append(kw)
        else:
            results.append(None)
            fallback_idx.append(i)
            fallback_texts.append(text)

    if fallback_texts:
        for i, res in zip(fallback_idx, get_tfidf().classify(fallback_texts)):
            results[i] = res

    return results


def write_batch(db: Session, updates: list[dict]) -> None:
    db.execute(
        text("""
            UPDATE reviews
            SET cleaned_text = :cleaned_text,
                primary_aspect = :primary_aspect,
                aspect_confidence = :aspect_confidence
            WHERE review_id = :review_id
        """),
        updates,
    )
    db.commit()


def enrich(sample: Optional[int] = None, batch_size: int = 500) -> None:
    verify_connection()
    db: Session = SessionLocal()

    try:
        pending = db.execute(
            text("SELECT COUNT(*) FROM reviews WHERE primary_aspect IS NULL")
        ).scalar()

        if pending == 0:
            logger.info("✓ All reviews already enriched. Nothing to do.")
            return

        logger.info(f"Reviews pending enrichment: {pending:,}")

        limit = f"LIMIT {sample}" if sample else ""
        rows  = db.execute(
            text(f"""
                SELECT review_id, review_text, review_summary
                FROM reviews WHERE primary_aspect IS NULL
                ORDER BY review_id {limit}
            """)
        ).fetchall()

        total = len(rows)
        logger.info(f"Fetched {total:,} reviews | batch_size={batch_size}")
        if sample:
            logger.info(f"Sample mode: {sample:,}")

        # Build TF-IDF classifier once before the loop
        get_tfidf()

        processed      = 0
        keyword_count  = 0
        tfidf_count    = 0
        t_start        = time.time()
        n_batches      = (total + batch_size - 1) // batch_size

        for batch_idx, start in enumerate(range(0, total, batch_size), 1):
            t0    = time.time()
            batch = rows[start : start + batch_size]
            texts = [clean_text(r.review_summary or r.review_text or "") for r in batch]

            aspect_results = classify_batch(texts)

            for _, conf in aspect_results:
                if conf == 0.75:
                    keyword_count += 1
                else:
                    tfidf_count += 1

            updates = [
                {
                    "review_id":         batch[i].review_id,
                    "cleaned_text":      texts[i],
                    "primary_aspect":    aspect_results[i][0],
                    "aspect_confidence": aspect_results[i][1],
                }
                for i in range(len(batch))
            ]
            write_batch(db, updates)

            processed += len(batch)
            elapsed    = time.time() - t_start
            rate       = processed / elapsed if elapsed > 0 else 1
            eta        = (total - processed) / rate

            logger.info(
                f"Batch {batch_idx}/{n_batches} | "
                f"{processed:,}/{total:,} ({100*processed/total:.1f}%) | "
                f"{time.time()-t0:.2f}s | "
                f"{rate:.0f} rev/s | "
                f"ETA: {eta/60:.1f}min"
            )

        elapsed = time.time() - t_start
        logger.info("=" * 55)
        logger.info("✓ Enrichment complete")
        logger.info(f"  Processed  : {processed:,} in {elapsed:.1f}s ({elapsed/60:.1f}min)")
        logger.info(f"  Speed      : {processed/elapsed:.0f} reviews/sec")
        logger.info(f"  Tier 1 (keyword) : {keyword_count:,} ({100*keyword_count/processed:.1f}%)")
        logger.info(f"  Tier 2 (TF-IDF)  : {tfidf_count:,}  ({100*tfidf_count/processed:.1f}%)")
        logger.info("=" * 55)

        dist = db.execute(text("""
            SELECT primary_aspect, COUNT(*) as cnt
            FROM reviews WHERE primary_aspect IS NOT NULL
            GROUP BY primary_aspect ORDER BY cnt DESC
        """)).fetchall()

        total_e = sum(r.cnt for r in dist)
        logger.info("Aspect distribution:")
        for r in dist:
            pct = 100 * r.cnt / total_e
            logger.info(f"  {r.primary_aspect:<15} {r.cnt:>8,}  {pct:5.1f}%  {'█' * int(pct/2)}")

        logger.info("Next step: run vector_store/vector_manager.py")

    finally:
        db.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enrich(sample=args.sample, batch_size=args.batch_size)
