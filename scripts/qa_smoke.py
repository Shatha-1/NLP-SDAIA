"""Lab 3B starter: run the 12-question QA smoke set.

Lab 3B has no QA fine-tuning script (unlike the classifier and NER labs) --
best_span() is exercised zero-shot against an off-the-shelf SQuAD2 checkpoint.

Note: the supplied data/eval/qa_smoke_set.json only contains answerable
questions (is_impossible=false for all 12 rows), so it cannot supply the
"3 unanswerable" half of the target on its own. The unanswerable examples
exist in the larger training fixture data/models/bayan_qa.json (100 of its
600 QAs are is_impossible=true), so the 12-question smoke set below is built
from that file instead: 9 unique answerable questions + 3 unique unanswerable
questions, deterministically selected.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span

CHECKPOINT = "deepset/roberta-base-squad2"
QA_DATA_PATH = Path("data/models/bayan_qa.json")
MAX_LENGTH = 384
NULL_THRESHOLD = 0.0
N_ANSWERABLE = 9
N_IMPOSSIBLE = 3


def _load_smoke_set(path: Path, n_answerable: int = N_ANSWERABLE, n_impossible: int = N_IMPOSSIBLE):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["data"]

    answerable, impossible = [], []
    seen_questions = set()
    for article in data:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa["question"] in seen_questions:
                    continue
                if qa.get("is_impossible"):
                    if len(impossible) < n_impossible:
                        impossible.append({"context": context, "question": qa["question"], "answers": []})
                        seen_questions.add(qa["question"])
                else:
                    if len(answerable) < n_answerable:
                        answerable.append(
                            {"context": context, "question": qa["question"], "answers": qa["answers"]}
                        )
                        seen_questions.add(qa["question"])
        if len(answerable) >= n_answerable and len(impossible) >= n_impossible:
            break
    return answerable + impossible


def _predict(tokenizer, model, question: str, context: str):
    enc = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    sequence_ids = enc.sequence_ids(0)
    offsets = [
        tuple(off) if sequence_ids[i] == 1 else None for i, off in enumerate(offset_mapping)
    ]

    with torch.no_grad():
        out = model(**enc)
    start_logits = out.start_logits[0].tolist()
    end_logits = out.end_logits[0].tolist()
    null_score = start_logits[0] + end_logits[0]  # [CLS] position

    result = best_span(start_logits, end_logits, offsets, null_score=null_score, null_threshold=NULL_THRESHOLD)
    if result["answer"] is None:
        return None
    a = result["answer"]
    return context[a["start_char"] : a["end_char"]]


def main():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForQuestionAnswering.from_pretrained(CHECKPOINT)
    model.eval()

    smoke_set = _load_smoke_set(QA_DATA_PATH)
    n_answerable = sum(1 for q in smoke_set if q["answers"])
    n_impossible = len(smoke_set) - n_answerable

    print(f"Checkpoint: {CHECKPOINT} (zero-shot; Lab 3B has no QA fine-tuning step)")
    print(f"Smoke set: {n_answerable} answerable + {n_impossible} unanswerable questions\n")

    correct_answerable = 0
    correct_null = 0
    for item in smoke_set:
        pred = _predict(tokenizer, model, item["question"], item["context"])
        if item["answers"]:
            gold = item["answers"][0]["text"]
            ok = pred is not None and gold.strip().lower() in pred.strip().lower()
            correct_answerable += int(ok)
        else:
            gold = None
            ok = pred is None
            correct_null += int(ok)
        print(f"[{'OK' if ok else 'MISS'}] Q: {item['question']!r} -> pred={pred!r} gold={gold!r}")

    print()
    print(f"Answerable correct:   {correct_answerable}/{n_answerable}")
    print(f"Unanswerable correct: {correct_null}/{n_impossible}")
    print("Record these numbers in BENCHMARKS.md under 'Lab 3 - Models'.")


if __name__ == "__main__":
    main()
