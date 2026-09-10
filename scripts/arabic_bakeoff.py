"""Lab 4 starter: compare Arabic-centric checkpoints by all/Gulf/MSA slices."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bayan.models.data import build_topic_dataset
from bayan.preprocessing.core import preprocess

# Day-2 multilingual incumbent: the already fine-tuned Lab 3A classifier.
INCUMBENT_CHECKPOINT_DIR = "artifacts/topic_classifier"

# MARBERT skipped: "optional if time allows" per the lab instructions.
CANDIDATES = {
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}
MAX_LENGTH = 64
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 2e-5


class TopicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=MAX_LENGTH
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def _slice_f1(model, tokenizer, texts, labels, device) -> float:
    if len(labels) == 0:
        return float("nan")
    loader = DataLoader(TopicDataset(texts, labels, tokenizer), batch_size=BATCH_SIZE)
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            gold.extend(y.tolist())
    return f1_score(gold, preds, average="macro")


def _fertility(tokenizer, texts) -> float:
    total_pieces = total_words = 0
    for t in texts:
        words = t.split()
        if not words:
            continue
        total_words += len(words)
        total_pieces += len(tokenizer.tokenize(t))
    return total_pieces / total_words if total_words else 0.0


def _evaluate(label, model, tokenizer, ar_test, label2id, device) -> dict:
    all_texts = ar_test["text"].map(preprocess)
    all_labels = ar_test["topic"].map(label2id)
    gulf = ar_test[ar_test["dialect_region"] == "Gulf"]
    msa = ar_test[ar_test["dialect_region"] == "MSA"]

    result = {
        "all": _slice_f1(model, tokenizer, all_texts, all_labels, device),
        "gulf": _slice_f1(model, tokenizer, gulf["text"].map(preprocess), gulf["topic"].map(label2id), device),
        "msa": _slice_f1(model, tokenizer, msa["text"].map(preprocess), msa["topic"].map(label2id), device),
        "ar_fertility": _fertility(tokenizer, all_texts),
    }
    print(
        f"{label:<24} all={result['all']:.4f}  Gulf={result['gulf']:.4f}  "
        f"MSA={result['msa']:.4f}  AR_fertility={result['ar_fertility']:.3f}"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--incumbent-dir",
        default=INCUMBENT_CHECKPOINT_DIR,
        help=(
            "Path to the already fine-tuned Lab 3A classifier artefact "
            "(artifacts/topic_classifier is .gitignored, so on a fresh clone "
            "point this at a Drive/local copy or retrain it first)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ds = build_topic_dataset()
    label_names = sorted(ds["train"]["topic"].unique())
    label2id = {label: i for i, label in enumerate(label_names)}

    ar_train = ds["train"][ds["train"]["lang"] == "ar"]
    # The frozen test split is 100% English (0 Arabic rows) in this dataset --
    # verified: ds["test"]["lang"].value_counts() == {"en": 1200}. Validation
    # is balanced (1200 ar / 1200 en) and unused for any tuning decision here,
    # so it's the only split that can actually answer an all/Gulf/MSA question.
    ar_test = ds["validation"][ds["validation"]["lang"] == "ar"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    incumbent_tok = AutoTokenizer.from_pretrained(args.incumbent_dir)
    incumbent_model = AutoModelForSequenceClassification.from_pretrained(args.incumbent_dir).to(device)
    results["multilingual incumbent (XLM-R)"] = _evaluate(
        "multilingual incumbent (XLM-R)", incumbent_model, incumbent_tok, ar_test, label2id, device
    )

    for label, checkpoint in CANDIDATES.items():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=len(label_names)
        ).to(device)

        train_texts = ar_train["text"].map(preprocess)
        train_labels = ar_train["topic"].map(label2id)
        train_loader = DataLoader(
            TopicDataset(train_texts, train_labels, tokenizer), batch_size=BATCH_SIZE, shuffle=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        model.train()
        start = time.time()
        for _ in range(EPOCHS):
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
        train_time = time.time() - start
        print(f"{label} fine-tuned on Arabic-only data in {train_time:.1f}s")

        results[label] = _evaluate(label, model, tokenizer, ar_test, label2id, device)

    incumbent_gulf = results["multilingual incumbent (XLM-R)"]["gulf"]
    for label in CANDIDATES:
        delta = (results[label]["gulf"] - incumbent_gulf) * 100
        print(f"{label} Gulf-slice delta vs incumbent: {delta:+.2f} macro-F1 points")

    print("\nRecord these numbers in BENCHMARKS.md 'Lab 4 - Arabic model bake-off' and DECISIONS.md#arabic-model.")
    return results


if __name__ == "__main__":
    main()
