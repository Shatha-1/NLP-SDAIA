"""Lab 3A starter: fine-tune the Bayan topic classifier."""
import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bayan.models.data import build_topic_dataset
from bayan.preprocessing.core import preprocess

CHECKPOINT = "xlm-roberta-base"  # Lab 1 tokenizer/checkpoint decision
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact (local path or mounted Drive path).",
    )
    return parser.parse_args()


def _evaluate(model, loader, device) -> float:
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            gold.extend(labels.tolist())
    return f1_score(gold, preds, average="macro")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = build_topic_dataset()
    label_names = sorted(ds["train"]["topic"].unique())
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=len(label_names), id2label=id2label, label2id=label2id
    )

    def make_dataset(split_name):
        split = ds[split_name]
        texts = split["text"].map(preprocess)
        labels = split["topic"].map(label2id)
        return TopicDataset(texts, labels, tokenizer)

    train_loader = DataLoader(make_dataset("train"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(make_dataset("validation"), batch_size=BATCH_SIZE)
    test_loader = DataLoader(make_dataset("test"), batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start = time.time()
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(train_loader)} loss={loss.item():.4f}")
        print(f"Epoch {epoch + 1} mean loss: {total_loss / len(train_loader):.4f}")
    train_time = time.time() - start

    val_f1 = _evaluate(model, val_loader, device)
    test_f1 = _evaluate(model, test_loader, device)

    print(f"Validation macro-F1:  {val_f1:.4f}")
    print(f"Frozen test macro-F1: {test_f1:.4f}")
    print(f"Train time: {train_time:.1f}s")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": CHECKPOINT,
                "validation_macro_f1": val_f1,
                "test_macro_f1": test_f1,
                "train_time_seconds": train_time,
            },
            f,
            indent=2,
        )

    print(f"Saved artefact to {output_dir}")
    print("Record these numbers in BENCHMARKS.md under 'Lab 3 - Models'.")


if __name__ == "__main__":
    main()
