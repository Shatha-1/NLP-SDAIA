# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
Measured on `data/raw/bayan_feedback.csv` (7,200 AR rows, 4,800 EN rows) via `python notebooks/01_tokenizer_audit.py`.

| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.153 | 1.510 | 27 | 25 | 0.45% |
| XLM-R | 1.672 | 1.434 | 21 | 23 | 0.00% |
| CAMeLBERT | 1.405 | 2.705 | 20 | 38 | 0.80% |
| DistilBERT | 4.527 | 1.298 | 47 | 21 | 0.22% |

Notes:
- CAMeLBERT is the most efficient tokenizer for Arabic (lowest AR fertility, 1.405) but the worst for English (2.705) — it's an Arabic-specialised vocabulary, so English text gets split into far more subwords.
- DistilBERT (English-only, uncased) is the mirror image: best on EN (1.298) but by far the worst on AR (4.527, plus it lowercases/strips diacritics, which is lossy for Arabic).
- XLM-R has the best balanced bilingual fertility (1.672 AR / 1.434 EN) and 0.00% AR UNK rate (byte-level BPE never needs `<unk>`), at the lowest combined p95 sequence length (21 AR / 23 EN).
- mBERT is a reasonable second choice, close to XLM-R but with a slightly higher AR fertility and a small but nonzero AR UNK rate (0.45%).

- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | | | |
| Topic classifier | macro-F1 | | | |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:
