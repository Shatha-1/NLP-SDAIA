"""Lab 7 + capstone: Bayan FastAPI service.

Integrates Labs 1-7: shared preprocessing, the Lab 7 INT8 topic classifier and
NER artefacts, and the Lab 5 two-stage bilingual search index.
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
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForTokenClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bayan.preprocessing.core import preprocess
from bayan.search.service import CaseSearch
from bayan.serving.canaries import CanaryFailure, run_startup_canaries

NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
torch.set_num_threads(NUM_THREADS)

# INT8 wins: Lab 7 measured 0.0000 paired quality tax (fp32 - INT8) for both the
# topic classifier and NER, so the quantised artefacts serve and the fp32
# artefacts are kept loaded only as the rollback/canary reference.
CLASSIFIER_SERVING_DIR = "artifacts/topic_classifier_int8"
CLASSIFIER_ROLLBACK_DIR = "artifacts/topic_classifier"
NER_SERVING_DIR = "artifacts/ner_int8"
NER_ROLLBACK_DIR = "artifacts/ner"
SEARCH_INDEX_PREFIX = "artifacts/search/case_index_v1"

app = FastAPI(title="Bayan — Bilingual Citizen-Feedback Intelligence Service")

_state: dict = {"canaries_ok": False, "canary_error": None}


def _onnx_session_options() -> ort.SessionOptions:
    # Measured both intra_op_num_threads=1 and =NUM_THREADS under the 16-
    # concurrent load test; NUM_THREADS (4) gave a lower p99 (63ms vs 132ms) --
    # see BENCHMARKS.md Lab 7 HTTP load-test note for both measurements.
    options = ort.SessionOptions()
    options.intra_op_num_threads = NUM_THREADS
    options.inter_op_num_threads = 1
    return options


def _build_classifier_predict_fn(model_dir: str, is_onnx: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if is_onnx:
        model = ORTModelForSequenceClassification.from_pretrained(model_dir, session_options=_onnx_session_options())
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


def _build_ner_predict_fn(model_dir: str, is_onnx: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if is_onnx:
        model = ORTModelForTokenClassification.from_pretrained(model_dir, session_options=_onnx_session_options())
    else:
        from transformers import AutoModelForTokenClassification

        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        model.eval()
    id2label = model.config.id2label

    def predict(text: str) -> list[dict]:
        words = preprocess(text).split()
        if not words:
            return []
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = logits.argmax(dim=-1)[0].tolist()
        word_ids = enc.word_ids()

        entities = []
        prev_word_id = None
        for wid, pid in zip(word_ids, pred_ids):
            if wid is None or wid == prev_word_id:
                prev_word_id = wid
                continue
            tag = id2label[pid]
            if tag != "O":
                entities.append({"type": tag.split("-", 1)[-1], "text": words[wid]})
            prev_word_id = wid
        return entities

    return predict


_classify_serving = _build_classifier_predict_fn(CLASSIFIER_SERVING_DIR, is_onnx=True)
_classify_rollback = _build_classifier_predict_fn(CLASSIFIER_ROLLBACK_DIR, is_onnx=False)
_extract_entities = _build_ner_predict_fn(NER_SERVING_DIR, is_onnx=True)

try:
    _case_search = CaseSearch(SEARCH_INDEX_PREFIX)
    _search_error = None
except Exception as exc:  # index missing/corrupt -- degrade /v1/search, not the whole service
    _case_search = None
    _search_error = str(exc)

try:
    run_startup_canaries(
        CLASSIFIER_SERVING_DIR,
        CLASSIFIER_ROLLBACK_DIR,
        predict_fn=lambda text: _classify_serving(text)["topic"],
        rollback_predict_fn=lambda text: _classify_rollback(text)["topic"],
    )
    _state["canaries_ok"] = True
except CanaryFailure as exc:
    _state["canary_error"] = str(exc)


def _require_canaries_ok():
    if not _state["canaries_ok"]:
        raise HTTPException(status_code=503, detail="Startup canaries failed; not serving traffic")


def _require_text(payload: dict) -> str:
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=422, detail="'text' field is required")
    return text


@app.get("/health")
def health():
    if not _state["canaries_ok"]:
        return {"status": "degraded", "canaries_ok": False, "error": _state["canary_error"]}
    return {
        "status": "ok",
        "canaries_ok": True,
        "serving_artefacts": {"classifier": CLASSIFIER_SERVING_DIR, "ner": NER_SERVING_DIR},
        "search_ready": _case_search is not None,
    }


@app.post("/v1/classify")
def classify(payload: dict):
    _require_canaries_ok()
    return _classify_serving(_require_text(payload))


@app.post("/v1/classify:batch")
def classify_batch(payload: dict):
    """Extension: batch classification for multiple feedback items in one call."""
    _require_canaries_ok()
    texts = payload.get("texts")
    if not texts or not isinstance(texts, list):
        raise HTTPException(status_code=422, detail="'texts' field (list) is required")
    return {"results": [_classify_serving(text) for text in texts]}


@app.post("/v1/entities")
def entities(payload: dict):
    _require_canaries_ok()
    return {"entities": _extract_entities(_require_text(payload))}


@app.post("/v1/search")
def search(payload: dict):
    _require_canaries_ok()
    if _case_search is None:
        raise HTTPException(status_code=503, detail=f"Search index unavailable: {_search_error}")
    query = _require_text(payload)
    k = int(payload.get("k", 5))
    return {"results": _case_search.search(query, k=k)}


@app.post("/v1/analyse")
def analyse(payload: dict):
    """One bilingual request -> classification + entities + similar historical cases."""
    _require_canaries_ok()
    text = _require_text(payload)
    result = {
        "classification": _classify_serving(text),
        "entities": _extract_entities(text),
    }
    if _case_search is not None:
        result["similar_cases"] = _case_search.search(text, k=int(payload.get("k", 5)))
    else:
        result["similar_cases"] = []
        result["search_error"] = _search_error
    return result
