"""Lab 5 starter: versioned FAISS index build."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess

DEFAULT_CASES_PATH = Path("data/search/bayan_cases.csv")
ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_encoder_cache: dict[str, SentenceTransformer] = {}


def _get_encoder(model_name: str) -> SentenceTransformer:
    if model_name not in _encoder_cache:
        _encoder_cache[model_name] = SentenceTransformer(model_name)
    return _encoder_cache[model_name]


def build_index(
    prefix: str,
    cases_path: Path = DEFAULT_CASES_PATH,
    model_name: str = ENCODER_MODEL,
    limit: int | None = None,
) -> dict:
    """Encode the case corpus, L2-normalise, build a FAISS index, and persist
    index + metadata + a manifest pinning model/preprocessing versions.
    """
    df = pd.read_csv(cases_path)
    if limit is not None:
        df = df.head(limit)

    texts = df["case_text"].map(preprocess).tolist()

    encoder = _get_encoder(model_name)
    embeddings = encoder.encode(
        texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    ).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, f"{prefix}.faiss")

    metadata = df[["case_id", "lang", "topic", "case_text", "resolution", "status"]].to_dict(
        orient="records"
    )
    with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    manifest = {
        "model": model_name,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": int(dim),
    }
    with open(f"{prefix}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
