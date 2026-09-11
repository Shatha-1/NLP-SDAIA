"""Lab 5 starter: labelled-query retrieval evaluation."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from bayan.search.index import build_index
from bayan.search.service import CaseSearch

INDEX_PREFIX = "artifacts/search/case_index_v1"
QUERIES_PATH = Path("data/search/bayan_queries.jsonl")
CASES_PATH = Path("data/search/bayan_cases.csv")
K = 10
CANDIDATES = 50
MIN_SCORE_CANDIDATES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]


def load_queries() -> list[dict]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_text_match_groups() -> dict[str, set]:
    """Maps every case_id to the full set of case_ids sharing its exact case_text.

    The corpus has heavy template duplication (20,000 rows, only 5,401 unique
    case_text values -- some templates repeat 300+ times), so a query's 3
    labelled `relevant_case_ids` often have dozens of textually-identical twins
    elsewhere in the corpus. Strict case_id recall unfairly penalises retrieving
    one of those twins instead of the exact labelled row. This "lenient" set
    lets us also report a fair, content-based recall alongside the strict one.
    """
    df = pd.read_csv(CASES_PATH)
    text_to_ids = defaultdict(set)
    for case_id, text in zip(df["case_id"], df["case_text"]):
        text_to_ids[text].add(case_id)
    id_to_text = dict(zip(df["case_id"], df["case_text"]))
    return {cid: text_to_ids[id_to_text[cid]] for cid in id_to_text}


def _expand_lenient(relevant_ids, text_match_groups) -> set:
    expanded = set()
    for cid in relevant_ids:
        expanded |= text_match_groups.get(cid, {cid})
    return expanded


def _recall(retrieved_ids, relevant_ids) -> float:
    return 1.0 if set(retrieved_ids) & set(relevant_ids) else 0.0


def _reciprocal_rank(retrieved_ids, relevant_ids) -> float:
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate(search: CaseSearch, queries, text_match_groups, use_rerank: bool, min_score: float) -> dict:
    recalls, mrrs, lenient_recalls, lenient_mrrs, topic_precisions = [], [], [], [], []
    recalls_by_lang = {"ar": [], "en": []}
    no_answer_correct = no_answer_total = 0
    latencies = []

    for q in queries:
        start = time.perf_counter()
        if use_rerank:
            results = search.search(q["query"], k=K, candidates=CANDIDATES, min_score=min_score)
        else:
            results = search.search_bi_encoder_only(q["query"], k=K)
        latencies.append(time.perf_counter() - start)
        retrieved_ids = [r["case_id"] for r in results]

        if q["no_answer"]:
            no_answer_total += 1
            is_empty = (not results) if use_rerank else (not results or results[0]["score"] < min_score)
            no_answer_correct += int(is_empty)
            continue

        lenient_relevant = _expand_lenient(q["relevant_case_ids"], text_match_groups)

        recalls.append(_recall(retrieved_ids, q["relevant_case_ids"]))
        mrrs.append(_reciprocal_rank(retrieved_ids, q["relevant_case_ids"]))
        lenient_recalls.append(_recall(retrieved_ids, lenient_relevant))
        lenient_mrrs.append(_reciprocal_rank(retrieved_ids, lenient_relevant))
        recalls_by_lang[q["lang"]].append(lenient_recalls[-1])
        if results:
            topic_precisions.append(sum(1 for r in results if r["topic"] == q["topic"]) / len(results))

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "topic_precision@10": avg(topic_precisions),
        "recall@10_strict": avg(recalls),
        "mrr@10_strict": avg(mrrs),
        "recall@10": avg(lenient_recalls),
        "mrr@10": avg(lenient_mrrs),
        "recall@10_ar": avg(recalls_by_lang["ar"]),
        "recall@10_en": avg(recalls_by_lang["en"]),
        "no_answer_correct": no_answer_correct,
        "no_answer_total": no_answer_total,
        "p50_latency_s": p50,
    }


def tune_min_score(search, queries, text_match_groups):
    print("Tuning min_score on the labelled query set (recall vs no-answer accuracy trade-off):")
    print(f"{'min_score':>10} {'recall@10':>10} {'no_answer_ok':>14}")
    best = None
    for candidate in MIN_SCORE_CANDIDATES:
        result = evaluate(search, queries, text_match_groups, use_rerank=True, min_score=candidate)
        no_answer_rate = result["no_answer_correct"] / result["no_answer_total"]
        print(f"{candidate:>10.2f} {result['recall@10']:>10.4f} {no_answer_rate:>13.1%}")
        # Prefer the lowest threshold that still gets every no-answer query right;
        # a threshold that lets false positives through is worse than one that's
        # slightly conservative on recall, for a "did we already resolve this" tool.
        if no_answer_rate == 1.0 and best is None:
            best = candidate
    return best if best is not None else MIN_SCORE_CANDIDATES[-1]


def main():
    index_path = Path(f"{INDEX_PREFIX}.faiss")
    if not index_path.exists():
        print("Building index over the full case corpus (a few minutes on CPU)...")
        manifest = build_index(prefix=INDEX_PREFIX)
        print(f"Index built: {manifest}")

    search = CaseSearch(INDEX_PREFIX)
    queries = load_queries()
    text_match_groups = build_text_match_groups()
    print(f"Loaded {len(queries)} labelled queries.\n")

    # A full sweep over MIN_SCORE_CANDIDATES already showed recall@10 and
    # no-answer accuracy are both flat across [0.02, 0.25] on this query set
    # (clean separation between answerable and no-answer scores), so 0.10 is
    # used directly here as a robust middle-of-the-range choice.
    tuned_min_score = 0.10
    print(f"min_score = {tuned_min_score} (see NOTES.md for the full sweep)\n")

    without_rerank = evaluate(search, queries, text_match_groups, use_rerank=False, min_score=tuned_min_score)
    with_rerank = evaluate(search, queries, text_match_groups, use_rerank=True, min_score=tuned_min_score)

    print("Without rerank (bi-encoder only):")
    print(
        f"  recall@10={without_rerank['recall@10']:.4f} (strict id: {without_rerank['recall@10_strict']:.4f})"
        f"  MRR@10={without_rerank['mrr@10']:.4f}"
    )
    print(f"  AR recall@10={without_rerank['recall@10_ar']:.4f}  EN recall@10={without_rerank['recall@10_en']:.4f}")
    print(f"  topic_precision@10={without_rerank['topic_precision@10']:.4f}")
    print(f"  p50 latency: {without_rerank['p50_latency_s'] * 1000:.1f} ms/query")

    print("\nWith cross-encoder rerank:")
    print(
        f"  recall@10={with_rerank['recall@10']:.4f} (strict id: {with_rerank['recall@10_strict']:.4f})"
        f"  MRR@10={with_rerank['mrr@10']:.4f}"
    )
    print(f"  AR recall@10={with_rerank['recall@10_ar']:.4f}  EN recall@10={with_rerank['recall@10_en']:.4f}")
    print(f"  topic_precision@10={with_rerank['topic_precision@10']:.4f}")
    print(f"  p50 latency: {with_rerank['p50_latency_s'] * 1000:.1f} ms/query")

    gap = with_rerank["recall@10_en"] - with_rerank["recall@10_ar"]
    print(f"\nCross-lingual recall@10 gap (EN - AR), with rerank: {gap:+.4f}")

    print(
        f"\nNo-answer empty-correct: {with_rerank['no_answer_correct']}/{with_rerank['no_answer_total']} "
        f"(min_score={tuned_min_score})"
    )

    print("\nRecord these numbers in BENCHMARKS.md under 'Lab 5 - Search'.")
    return {"without_rerank": without_rerank, "with_rerank": with_rerank, "min_score": tuned_min_score}


if __name__ == "__main__":
    main()
