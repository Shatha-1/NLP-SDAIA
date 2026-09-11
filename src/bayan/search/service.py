"""Lab 5 starter: two-stage bilingual case search."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CaseSearch:
    def __init__(self, prefix: str):
        manifest_path = Path(f"{prefix}_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("model", "preproc_version", "n_vectors", "dim"):
            assert key in manifest, f"manifest missing required key: {key}"

        self.index = faiss.read_index(f"{prefix}.faiss")
        assert self.index.ntotal == manifest["n_vectors"], "manifest/index vector count mismatch"
        assert self.index.d == manifest["dim"], "manifest/index dimension mismatch"

        if manifest["preproc_version"] != PREPROC_VERSION:
            raise ValueError(
                f"Index was built with preprocessing v{manifest['preproc_version']}, "
                f"but the running code is v{PREPROC_VERSION}. Rebuild the index."
            )

        with open(f"{prefix}_meta.json", encoding="utf-8") as f:
            self.metadata = json.load(f)
        assert len(self.metadata) == manifest["n_vectors"], "manifest/metadata count mismatch"

        self.manifest = manifest
        self.encoder = SentenceTransformer(manifest["model"])
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def search_bi_encoder_only(self, query: str, k: int = 10) -> list[dict]:
        """Bi-encoder retrieval with no cross-encoder rerank, for ablation."""
        normalized_query = preprocess(query)
        query_vec = self._embed_query(normalized_query)
        scores, indices = self.index.search(query_vec, min(k, self.index.ntotal))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append({**self.metadata[idx], "score": float(score)})
        return results

    def search(self, query: str, k: int = 5, candidates: int = 50, min_score: float = 0.10):
        """Bi-encoder retrieval + cross-encoder rerank, with an honest empty result
        when the best reranked candidate scores below min_score.

        The reranker (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) outputs raw,
        unbounded logits, not a [0,1] score -- a sigmoid is applied here so
        min_score is an interpretable, tunable probability-like threshold. Even
        genuinely relevant short-text pairs on this domain only reach ~0.10-0.20
        after sigmoid (see notebooks/05_retrieval_eval.py for the empirical sweep
        that picked the default above), so this is NOT a conventional 0.5-ish
        similarity cutoff.
        """
        normalized_query = preprocess(query)
        query_vec = self._embed_query(normalized_query)

        n_candidates = min(candidates, self.index.ntotal)
        _, indices = self.index.search(query_vec, n_candidates)
        candidate_indices = [i for i in indices[0] if i != -1]
        if not candidate_indices:
            return []

        pairs = [(normalized_query, self.metadata[i]["case_text"]) for i in candidate_indices]
        ce_scores = _sigmoid(np.array(self.reranker.predict(pairs)))

        ranked = sorted(zip(candidate_indices, ce_scores), key=lambda x: x[1], reverse=True)

        if not ranked or ranked[0][1] < min_score:
            return []

        results = []
        for idx, score in ranked[:k]:
            if score < min_score:
                break
            case = self.metadata[idx]
            results.append({**case, "score": float(score)})
        return results
