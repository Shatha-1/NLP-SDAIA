"""Lab 7 starter: ONNX export and dynamic INT8 quantisation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
    ORTQuantizer,
)
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from bayan.evaluation.bootstrap import paired_bootstrap_diff
from bayan.models.data import build_topic_dataset
from bayan.preprocessing.core import preprocess

CLASSIFIER_FP32_DIR = "artifacts/topic_classifier"
CLASSIFIER_ONNX_DIR = "artifacts/topic_classifier_onnx"
CLASSIFIER_INT8_DIR = "artifacts/topic_classifier_int8"

NER_FP32_DIR = "artifacts/ner"
NER_ONNX_DIR = "artifacts/ner_onnx"
NER_INT8_DIR = "artifacts/ner_int8"


def export_and_quantize(fp32_dir: str, onnx_dir: str, int8_dir: str, model_cls):
    """Export a fp32 HF model to ONNX, then dynamically quantise it to INT8.
    The fp32 artefact is left untouched at fp32_dir as the rollback."""
    tokenizer = AutoTokenizer.from_pretrained(fp32_dir)

    onnx_model = model_cls.from_pretrained(fp32_dir, export=True)
    onnx_model.save_pretrained(onnx_dir)
    tokenizer.save_pretrained(onnx_dir)

    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=int8_dir, quantization_config=qconfig)
    tokenizer.save_pretrained(int8_dir)


def _classifier_predict_fn(model_dir: str, is_onnx: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if is_onnx:
        model = ORTModelForSequenceClassification.from_pretrained(model_dir)
    else:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
    id2label = model.config.id2label

    def predict(text: str) -> str:
        inputs = tokenizer(preprocess(text), return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = model(**inputs).logits
        return id2label[int(logits.argmax(dim=-1))]

    return predict


def evaluate_classifier_quality_tax():
    print("\n=== Classifier: fp32 vs ONNX-fp32 vs INT8 quality tax ===")
    ds = build_topic_dataset()
    val = ds["validation"]
    texts = val["text"].tolist()
    y_true = val["topic"].tolist()

    variants = {
        "fp32 (torch)": _classifier_predict_fn(CLASSIFIER_FP32_DIR, is_onnx=False),
        "ONNX fp32": _classifier_predict_fn(CLASSIFIER_ONNX_DIR, is_onnx=True),
        "ONNX INT8": _classifier_predict_fn(CLASSIFIER_INT8_DIR, is_onnx=True),
    }

    preds = {}
    for name, predict_fn in variants.items():
        preds[name] = [predict_fn(t) for t in texts]
        macro_f1 = f1_score(y_true, preds[name], average="macro")
        print(f"  {name:<14} macro-F1 = {macro_f1:.4f}")

    fp32_correct = [int(p == t) for p, t in zip(preds["fp32 (torch)"], y_true)]
    int8_correct = [int(p == t) for p, t in zip(preds["ONNX INT8"], y_true)]
    delta, lo, hi = paired_bootstrap_diff(fp32_correct, int8_correct)
    print(f"  Paired quality tax (fp32 - INT8 accuracy): {delta:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return delta, lo, hi


def _ner_predict_fn(model_dir: str, is_onnx: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if is_onnx:
        model = ORTModelForTokenClassification.from_pretrained(model_dir)
    else:
        from transformers import AutoModelForTokenClassification

        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        model.eval()
    id2label = model.config.id2label

    def predict(words: list[str]) -> list[str]:
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=32, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = logits.argmax(dim=-1)[0].tolist()
        word_ids = enc.word_ids()
        tags = []
        prev = None
        for wid, pid in zip(word_ids, pred_ids):
            if wid is None or wid == prev:
                continue
            tags.append(id2label[pid])
            prev = wid
        return tags

    return predict


def evaluate_ner_quality_tax():
    print("\n=== NER: fp32 vs ONNX-fp32 vs INT8 quality tax ===")
    from train_ner import read_conll, split_sentences

    sentences = read_conll(Path("data/models/bayan_ner.conll"))
    _, _, test_sents = split_sentences(sentences)

    variants = {
        "fp32 (torch)": _ner_predict_fn(NER_FP32_DIR, is_onnx=False),
        "ONNX fp32": _ner_predict_fn(NER_ONNX_DIR, is_onnx=True),
        "ONNX INT8": _ner_predict_fn(NER_INT8_DIR, is_onnx=True),
    }

    from seqeval.metrics import f1_score as seqeval_f1

    all_preds = {}
    gold = [s["tags"] for s in test_sents]
    for name, predict_fn in variants.items():
        preds = [predict_fn(s["words"]) for s in test_sents]
        preds = [p[: len(g)] + ["O"] * max(0, len(g) - len(p)) for p, g in zip(preds, gold)]
        all_preds[name] = preds
        print(f"  {name:<14} entity-F1 = {seqeval_f1(gold, preds):.4f}")

    fp32_exact = [int(p == g) for p, g in zip(all_preds["fp32 (torch)"], gold)]
    int8_exact = [int(p == g) for p, g in zip(all_preds["ONNX INT8"], gold)]
    delta, lo, hi = paired_bootstrap_diff(fp32_exact, int8_exact)
    print(f"  Paired quality tax (fp32 - INT8, sentence-exact-match): {delta:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return delta, lo, hi


def main():
    print("Exporting classifier to ONNX + INT8...")
    export_and_quantize(CLASSIFIER_FP32_DIR, CLASSIFIER_ONNX_DIR, CLASSIFIER_INT8_DIR, ORTModelForSequenceClassification)
    classifier_tax = evaluate_classifier_quality_tax()

    print("\nExporting NER to ONNX + INT8...")
    export_and_quantize(NER_FP32_DIR, NER_ONNX_DIR, NER_INT8_DIR, ORTModelForTokenClassification)
    ner_tax = evaluate_ner_quality_tax()

    print("\nRecord these numbers in BENCHMARKS.md under 'Lab 7 - Optimisation ladder' "
          "and the quantisation decision in DECISIONS.md#quantisation-split.")
    return {"classifier_tax": classifier_tax, "ner_tax": ner_tax}


if __name__ == "__main__":
    main()
