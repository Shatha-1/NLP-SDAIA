"""Lab 7 + capstone starter: Bayan FastAPI service.

The final service should integrate the outputs of Labs 1-7. Keep this file as
orchestration; reusable logic belongs in the package modules.
"""
import os

# Pin the thread count before torch/onnxruntime are imported anywhere below.
# Without this, each concurrent request's inference call is free to spin up
# threads across all cores, and under the 16-concurrent load test that causes
# CPU oversubscription and a much worse tail latency than single-request
# benchmarking shows (see BENCHMARKS.md Lab 7 -- this was a real, measured gap).
os.environ.setdefault("OMP_NUM_THREADS", "4")

import onnxruntime as ort
import torch
from fastapi import FastAPI, HTTPException
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bayan.preprocessing.core import preprocess
from bayan.serving.canaries import CanaryFailure, run_startup_canaries

NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
torch.set_num_threads(NUM_THREADS)

# INT8 wins: Lab 7 measured 0.0000 paired quality tax (fp32 - INT8) for both the
# topic classifier and NER, so the quantised artefact serves and the fp32
# artefact is kept loaded only as the rollback/canary reference.
CLASSIFIER_SERVING_DIR = "artifacts/topic_classifier_int8"
CLASSIFIER_ROLLBACK_DIR = "artifacts/topic_classifier"

app = FastAPI(title="Bayan — Bilingual Citizen-Feedback Intelligence Service")

_state: dict = {"canaries_ok": False, "canary_error": None}


def _build_classifier_predict_fn(model_dir: str, is_onnx: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if is_onnx:
        # Measured both intra_op_num_threads=1 and =NUM_THREADS under the 16-
        # concurrent load test; NUM_THREADS (4) gave a lower p99 (63ms vs 132ms)
        # -- see BENCHMARKS.md Lab 7 HTTP load-test note for both measurements.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = NUM_THREADS
        session_options.inter_op_num_threads = 1
        model = ORTModelForSequenceClassification.from_pretrained(model_dir, session_options=session_options)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
    id2label = model.config.id2label

    def predict(text: str) -> dict:
        inputs = tokenizer(preprocess(text), return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = logits.softmax(dim=-1)[0]
        pred_id = int(probs.argmax())
        return {"topic": id2label[pred_id], "confidence": float(probs[pred_id])}

    return predict


_serving_predict = _build_classifier_predict_fn(CLASSIFIER_SERVING_DIR, is_onnx=True)
_rollback_predict = _build_classifier_predict_fn(CLASSIFIER_ROLLBACK_DIR, is_onnx=False)

try:
    run_startup_canaries(
        CLASSIFIER_SERVING_DIR,
        CLASSIFIER_ROLLBACK_DIR,
        predict_fn=lambda text: _serving_predict(text)["topic"],
        rollback_predict_fn=lambda text: _rollback_predict(text)["topic"],
    )
    _state["canaries_ok"] = True
except CanaryFailure as exc:
    _state["canary_error"] = str(exc)


@app.get("/health")
def health():
    if not _state["canaries_ok"]:
        return {"status": "degraded", "canaries_ok": False, "error": _state["canary_error"]}
    return {"status": "ok", "canaries_ok": True, "serving_artefact": CLASSIFIER_SERVING_DIR}


@app.post("/v1/classify")
def classify(payload: dict):
    if not _state["canaries_ok"]:
        raise HTTPException(status_code=503, detail="Startup canaries failed; not serving traffic")
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=422, detail="'text' field is required")
    return _serving_predict(text)


@app.post("/v1/entities")
def entities(payload: dict):
    # TODO(Capstone): shared preprocessing/Arabic segmentation -> NER -> case fields.
    raise NotImplementedError("Wire the NER artefact")


@app.post("/v1/search")
def search(payload: dict):
    # TODO(Capstone): Lab 5 two-stage bilingual search.
    raise NotImplementedError("Wire the semantic-search component")


@app.post("/v1/analyse")
def analyse(payload: dict):
    # TODO(Capstone): one bilingual request -> classification + entities + similar cases.
    raise NotImplementedError("Assemble the Bayan capstone service")
