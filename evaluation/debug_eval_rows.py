import functools
import json
import logging

from agents.orchestrator import run_pipeline
from agents.vector_agent import extract_intent
from pipeline.query_rewriter import multi_query_retrieve
from pipeline.reranker import rerank
from pipeline.retriever import hybrid_search

logging.basicConfig(level=logging.WARNING)  # quiet down noisy INFO logs for readability

TEST_SET_PATH = "evaluation/test_set.json"
N_ROWS_TO_INSPECT = 10


def inspect_row(item: dict) -> None:
    question = item["question"]
    ground_truth = item["ground_truth"]

    intent = extract_intent(question)
    retrieve_fn = functools.partial(hybrid_search, category=intent["category"], aspect=intent["aspect"])
    candidates = multi_query_retrieve(question, retrieve_fn, top_k=15)
    final = rerank(question, candidates, top_k=5)

    pipeline_result = run_pipeline(question)

    print("=" * 90)
    print(f"QUESTION:        {question}")
    print(f"GROUND TRUTH:    {ground_truth}")
    print(f"SOURCE REVIEW ID (expected match): {item['source_review_id']}")
    print(f"INTENT EXTRACTED: category={intent['category']!r}, aspect={intent['aspect']!r}")
    print(f"ITEM'S TRUE category/aspect:        category={item['category']!r}, aspect={item['aspect']!r}")
    print("-" * 90)
    print("TOP RETRIEVED CONTEXTS (after rerank):")

    found_source = False
    for rank, r in enumerate(final, 1):
        review_id = r["payload"].get("review_id")
        is_source = review_id == item["source_review_id"]
        found_source = found_source or is_source
        marker = "  <-- MATCHES SOURCE REVIEW" if is_source else ""
        text = (r["payload"].get("review_text") or r["payload"].get("review_summary", ""))[:120]
        print(f"  {rank}. [rerank={r.get('rerank_score', 0):.2f}] {text}{marker}")

    print("-" * 90)
    print(f"Source review found in top-5 retrieved? {'YES' if found_source else 'NO'}")
    print(f"GENERATED ANSWER: {pipeline_result['answer'][:300]}")
    print()


def main():
    with open(TEST_SET_PATH, "r") as f:
        test_set = json.load(f)

    sample = test_set[:N_ROWS_TO_INSPECT]

    match_count = 0
    for item in sample:
        inspect_row(item)

    print("=" * 90)
    print(f"Inspected {len(sample)} rows. Scroll up to review each one.")


if __name__ == "__main__":
    main()