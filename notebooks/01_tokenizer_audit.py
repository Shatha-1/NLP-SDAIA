"""Lab 1 starter: audit four tokenizer candidates on Bayan AR/EN text."""
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    """Total subword pieces / whitespace words across texts."""
    total_pieces = 0
    total_words = 0
    for text in texts:
        words = text.split()
        if not words:
            continue
        total_words += len(words)
        total_pieces += len(tokenizer.tokenize(text))
    return total_pieces / total_words if total_words else 0.0


def _sequence_lengths(tokenizer, texts) -> list[int]:
    return [len(tokenizer.encode(text, add_special_tokens=True)) for text in texts]


def _unk_rate(tokenizer, texts) -> float:
    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        return 0.0
    total = 0
    unk = 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        total += len(ids)
        unk += sum(1 for i in ids if i == unk_id)
    return unk / total if total else 0.0


def main():
    df = pd.read_csv(DATA)
    ar_texts = df.loc[df["lang"] == "ar", "text"].astype(str).tolist()
    en_texts = df.loc[df["lang"] == "en", "text"].astype(str).tolist()

    print(f"AR rows: {len(ar_texts)} | EN rows: {len(en_texts)}")
    header = f"{'Tokenizer':<12}{'AR fert':>10}{'EN fert':>10}{'AR p95':>10}{'EN p95':>10}{'AR UNK%':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for checkpoint, label in CANDIDATES.items():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        ar_fert = fertility(tokenizer, ar_texts)
        en_fert = fertility(tokenizer, en_texts)
        ar_p95 = float(np.percentile(_sequence_lengths(tokenizer, ar_texts), 95))
        en_p95 = float(np.percentile(_sequence_lengths(tokenizer, en_texts), 95))
        ar_unk = _unk_rate(tokenizer, ar_texts) * 100

        results.append(
            {
                "checkpoint": checkpoint,
                "label": label,
                "ar_fertility": ar_fert,
                "en_fertility": en_fert,
                "ar_p95_len": ar_p95,
                "en_p95_len": en_p95,
                "ar_unk_rate_pct": ar_unk,
            }
        )
        print(
            f"{label:<12}{ar_fert:>10.3f}{en_fert:>10.3f}{ar_p95:>10.1f}{en_p95:>10.1f}{ar_unk:>9.2f}%"
        )

    print("\nCopy the rows above into BENCHMARKS.md under 'Lab 1 - Tokenizer audit'.")
    return results


if __name__ == "__main__":
    main()
