"""Lab 7 starter: honest p50/p99 benchmark harness over production length mix."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from bayan.preprocessing.core import preprocess

BENCH_MIX_PATH = Path("data/serving/bench_mix.npy")
CLASSIFIER_DIR = "artifacts/topic_classifier"
N_WARMUP = 20
N_MEASURE = 500


def benchmark(predict_fn, texts, *, n_warmup: int = N_WARMUP, n_measure: int = N_MEASURE) -> dict:
    """Single-request latency benchmark: warm-up, then measure p50/p99 over
    n_measure requests drawn from the production length mix."""
    for text in texts[:n_warmup]:
        predict_fn(text)

    sample = texts[n_warmup : n_warmup + n_measure]
    latencies = []
    for text in sample:
        start = time.perf_counter()
        predict_fn(text)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p99 = latencies[min(int(n * 0.99), n - 1)]
    return {"p50_ms": p50, "p99_ms": p99, "n": n}


def _make_torch_predict_fn(model_dir: str, max_length: int, dynamic_padding: bool):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

    padding = "longest" if dynamic_padding else "max_length"

    def predict(text: str):
        inputs = tokenizer(
            preprocess(text), return_tensors="pt", truncation=True, max_length=max_length, padding=padding
        )
        with torch.no_grad():
            model(**inputs)

    return predict


def _make_onnx_predict_fn(model_dir: str, max_length: int):
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ORTModelForSequenceClassification.from_pretrained(model_dir)

    def predict(text: str):
        inputs = tokenizer(
            preprocess(text), return_tensors="pt", truncation=True, max_length=max_length, padding="longest"
        )
        model(**inputs)

    return predict


def main():
    texts = np.load(BENCH_MIX_PATH, allow_pickle=True).tolist()
    print(f"Loaded {len(texts)} production-mix texts. OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

    rungs = [
        ("fp32 torch @512 padded", _make_torch_predict_fn(CLASSIFIER_DIR, 512, dynamic_padding=False)),
        ("fp32 torch @128 dynamic", _make_torch_predict_fn(CLASSIFIER_DIR, 128, dynamic_padding=True)),
        ("ONNX fp32 @128", _make_onnx_predict_fn("artifacts/topic_classifier_onnx", 128)),
        ("ONNX INT8 @128", _make_onnx_predict_fn("artifacts/topic_classifier_int8", 128)),
    ]

    print(f"\n{'Rung':<28}{'p50 (ms)':>12}{'p99 (ms)':>12}")
    for name, predict_fn in rungs:
        result = benchmark(predict_fn, texts)
        print(f"{name:<28}{result['p50_ms']:>12.2f}{result['p99_ms']:>12.2f}")

    print("\nRecord these rows in BENCHMARKS.md under 'Lab 7 - Optimisation ladder'.")


if __name__ == "__main__":
    main()
