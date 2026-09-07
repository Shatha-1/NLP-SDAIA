"""Lab 2 starter: parameter accounting for mBERT and CAMeLBERT."""
from transformers import AutoModel

BUCKET_ORDER = ["embeddings", "attention", "ffn", "norms", "pooler", "other"]


def _bucket(name: str) -> str:
    """Classify a BERT-style parameter name into an accounting bucket.

    Order matters: LayerNorm is pulled into its own bucket even when it lives
    inside embeddings/attention/ffn blocks, and attention.output.dense must be
    caught by the "attention" check before the generic "output.dense" FFN check.
    """
    if "pooler" in name:
        return "pooler"
    if "LayerNorm" in name:
        return "norms"
    if "embeddings" in name:
        return "embeddings"
    if "attention" in name:
        return "attention"
    if "intermediate" in name or "output.dense" in name:
        return "ffn"
    return "other"


def audit(checkpoint: str) -> dict:
    model = AutoModel.from_pretrained(checkpoint)

    buckets = {b: 0 for b in BUCKET_ORDER}
    for name, param in model.named_parameters():
        buckets[_bucket(name)] += param.numel()

    total = sum(buckets.values())
    bucket_pct = {b: (n / total * 100 if total else 0.0) for b, n in buckets.items()}

    return {
        "checkpoint": checkpoint,
        "total_params": total,
        "buckets": buckets,
        "bucket_pct": bucket_pct,
        "vocab_size": model.config.vocab_size,
    }


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        result = audit(ckpt)
        print(f"\n{ckpt}")
        print(f"  vocab_size:   {result['vocab_size']:,}")
        print(f"  total params: {result['total_params']:,}")
        for b in BUCKET_ORDER:
            n = result["buckets"][b]
            pct = result["bucket_pct"][b]
            print(f"  {b:<12} {n:>12,}  ({pct:5.1f}%)")
