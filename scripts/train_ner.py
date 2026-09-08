"""Lab 3B starter: fine-tune token classification with correct alignment."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from bayan.models.ner import align_labels

CHECKPOINT = "xlm-roberta-base"  # Lab 1 tokenizer/checkpoint decision
DATA_PATH = Path("data/models/bayan_ner.conll")
MAX_LENGTH = 32
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 3e-5
SEED = 42


def read_conll(path: Path) -> list[dict]:
    """Parse tab-separated word/tag CoNLL, blank-line separated sentences.

    A "word" here may itself contain an internal space (e.g. a two-word
    service name) — it is still one BIO-tagged unit in this dataset.
    """
    text = path.read_text(encoding="utf-8").strip()
    sentences = []
    for block in text.split("\n\n"):
        words, tags = [], []
        for line in block.strip().split("\n"):
            word, tag = line.split("\t")
            words.append(word)
            tags.append(tag)
        sentences.append({"words": words, "tags": tags})
    return sentences


def split_sentences(sentences: list[dict], seed: int = SEED):
    shuffled = sentences[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


class NerDataset(Dataset):
    def __init__(self, sentences, tokenizer, tag2id):
        self.encodings = []
        self.labels = []
        for sent in sentences:
            enc = tokenizer(
                sent["words"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
            )
            word_ids = enc.word_ids()
            word_labels = [tag2id[t] for t in sent["tags"]]
            self.encodings.append({k: v for k, v in enc.items()})
            self.labels.append(align_labels(word_ids, word_labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.encodings[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact (local path or mounted Drive path).",
    )
    return parser.parse_args()


def _predict(model, loader, device, id2label):
    model.eval()
    pred_tags, gold_tags = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1).cpu()
            for pred_row, gold_row in zip(preds.tolist(), labels.tolist()):
                p_seq, g_seq = [], []
                for p, g in zip(pred_row, gold_row):
                    if g == -100:
                        continue
                    p_seq.append(id2label[p])
                    g_seq.append(id2label[g])
                pred_tags.append(p_seq)
                gold_tags.append(g_seq)
    return pred_tags, gold_tags


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = read_conll(DATA_PATH)
    train_sents, val_sents, test_sents = split_sentences(sentences)

    tag_set = sorted({tag for sent in sentences for tag in sent["tags"]})
    tag2id = {tag: i for i, tag in enumerate(tag_set)}
    id2label = {i: tag for tag, i in tag2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT, num_labels=len(tag_set), id2label=id2label, label2id=tag2id
    )

    train_ds = NerDataset(train_sents, tokenizer, tag2id)
    val_ds = NerDataset(val_sents, tokenizer, tag2id)
    test_ds = NerDataset(test_sents, tokenizer, tag2id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

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

    val_preds, val_gold = _predict(model, val_loader, device, id2label)
    test_preds, test_gold = _predict(model, test_loader, device, id2label)

    val_f1 = seqeval_f1(val_gold, val_preds)
    test_f1 = seqeval_f1(test_gold, test_preds)

    print(f"Validation entity-F1:  {val_f1:.4f}")
    print(f"Frozen test entity-F1: {test_f1:.4f}")
    print(f"Train time: {train_time:.1f}s")
    print("\nTest set seqeval report:")
    print(seqeval_report(test_gold, test_preds, zero_division=0))

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": CHECKPOINT,
                "validation_entity_f1": val_f1,
                "test_entity_f1": test_f1,
                "train_time_seconds": train_time,
                "labels": tag_set,
            },
            f,
            indent=2,
        )

    print(f"Saved artefact to {output_dir}")
    print("Record these numbers in BENCHMARKS.md under 'Lab 3 - Models'.")


if __name__ == "__main__":
    main()
