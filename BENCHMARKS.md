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

## Lab 2 — Transformer anatomy
- `attention()` vs `F.scaled_dot_product_attention`: match at atol=1e-6 → **True**
- Attention weight matrix rows sum to 1 (valid softmax distribution): **True**
- `MultiHeadAttention` shape check (batch=2, seq=5, d_model=16, heads=4): input `(2, 5, 16)` → output `(2, 5, 16)` → **passed**
- Causal mask → attention matrix lower-triangular: **True** (decoder-style / GPT-family attention)

| Checkpoint | Total params | Embeddings % | Attention % | FFN % | Vocab size |
|---|---:|---:|---:|---:|---:|
| mBERT | 177,853,440 | 51.8% | 15.9% | 31.9% | 119,547 |
| CAMeLBERT | 109,081,344 | 21.5% | 26.0% | 52.0% | 30,000 |

**Pad-leak diagnosis** (mBERT, last layer, mean attention mass on `[PAD]` positions):
| Run | Mean attention mass on [PAD] |
|---|---:|
| WITH `attention_mask` | 0.00000 |
| WITHOUT `attention_mask` | 0.02694 |

Skipping `attention_mask` leaks ~2.7% of attention weight onto `[PAD]` tokens — silent quality degradation, no crash. See `NOTES.md` for the full head-by-head [CLS]/[SEP] sink breakdown.

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 | ~1s |
| Topic classifier (XLM-R, 1 epoch, CPU) | macro-F1 | 1.0000 | 0.7993 | 4563.1s (~76 min, CPU) |
| Topic classifier (XLM-R, 3 epochs, GPU) | macro-F1 | 1.0000 | **1.0000** | 314.1s (~5.2 min, T4 GPU) |
| NER (XLM-R, 1 epoch) | entity-F1 | 1.0000 | 1.0000 | 635.6s (~10.6 min, CPU) |
| QA (deepset/roberta-base-squad2, zero-shot) | span/null smoke | 9/9 answerable, 3/3 null | n/a (frozen at 12/12) | n/a (no training) |

**Topic classifier retrain (closes the val/test gap):** the 1-epoch CPU run above left a real 20-point val/test gap (see the note below). Retraining for 3 epochs on a Colab T4 GPU — same code, same data, same 70/20/10 split — brought frozen test macro-F1 up to a perfect **1.0000**, matching the TF-IDF baseline and validation. Final training loss dropped from 0.017 (epoch 1) to 0.0043 (epoch 3), confirming the 1-epoch run simply hadn't converged yet rather than there being a deeper problem with the data or model. This run now meets the letter of the Lab 3A target as well as the spirit (macro-F1 ties the baseline; it cannot mathematically exceed it by +0.08 since the baseline is already a perfect 1.0).

**Correction — the frozen test split is English-only:** discovered while building the Lab 4 bake-off that `bayan_feedback.csv`'s `split="test"` rows are 100% English (0 Arabic rows); `validation` is the balanced split (1,200 ar / 1,200 en). So every "frozen test macro-F1" above was measured on English text only, not bilingually. See `NOTES.md` for the full per-split language table and why this doesn't invalidate the numbers (they're still real, just narrower in scope than originally assumed).

**NER note:** all 4 entity types (DATE, LOCATION, REFERENCE, SERVICE) hit perfect precision/recall/F1 on the frozen test split — comfortably above the ≥0.80 target. This dataset's entities follow a small number of fixed sentence templates (e.g. "بلاغ عن X في Y بتاريخ Z مرجعه W"), which makes span boundaries very regular and easy for the model to learn; ORGANISATION does not appear in `bayan_ner.conll` at all (only DATE/LOCATION/REFERENCE/SERVICE + O are present in the data), so it isn't in this evaluation.

**Note on the topic classifier val/test gap:** validation macro-F1 hit a perfect 1.0000 (final training loss 0.017) but frozen test macro-F1 was only 0.7993 — a real, measured 20-point gap, not a copy-paste error. Working theory: with only 3,820 unique underlying texts recycled across the 12,000-row dataset, many validation sentences are near-duplicates of training sentences (same template, different citizen/date), so the model could partly memorize its way to a perfect validation score after just one epoch. The frozen test split apparently contains a somewhat harder/less-duplicated mix that one epoch wasn't enough to generalise to, whereas TF-IDF's much simpler keyword-based decision boundary generalised to both splits equally. This is reported as measured, not smoothed over — a case study for Lab 6's "don't approve by eyeballing a single split" principle.

**Note on the TF-IDF baseline:** `bayan_feedback.csv` only has 3,820 unique underlying texts spread across 12,000 rows (repeated templates across different citizens/dates), and each `topic` uses very distinct, non-overlapping vocabulary (e.g. "water leak" only ever appears for `water`, "ترخيص" only for `licensing`). That makes topic classification trivial from bag-of-words alone — the baseline reaches a perfect 1.0000 macro-F1 on both validation and the frozen test split, well above the course's "~0.71" reference (which was measured on the real, messier course corpus, not this synthetic reconstruction). This is a real, honest measurement, not a bug — but it also means the transformer classifier has no headroom to clear "+0.08 over baseline"; the achievable target here is to *match* 1.0 while demonstrating the fine-tuning pipeline works, not to beat an already-perfect baseline.

## Lab 4 — Clitic segmentation + NER LOCATION recall
| Run | Data | entity-F1 (test) | LOCATION recall (test) |
|---|---|---:|---:|
| Day-2 baseline | `bayan_ner.conll` | 1.0000 | 1.0000 |
| + segmentation | `bayan_ner_segmented.conll` | 1.0000 | 1.0000 |

**LOCATION recall delta: 0.00 points**, not the target +4. This isn't a bug — the Day-2 NER baseline was already a perfect 1.0000 on every entity type (see Lab 3B), so there is no headroom left for segmentation to improve on; recall is capped at 1.0 either way. The supplied `bayan_ner_segmented.conll` also never actually touches LOCATION-tagged tokens (verified: 0/4,000 sentences have a different LOCATION line between the two files) — the only word it clitic-splits is the non-entity `"مرجعه"` → `"مرجع"` + `"ه"`. So even setting the ceiling effect aside, this specific dataset's LOCATION entities never had an attached-clitic problem for segmentation to fix. Same pattern as the Lab 3A baseline ceiling: the synthetic data is easy enough that several targets calibrated for the real, messier course corpus aren't measurable here.

## Lab 4 — Arabic model bake-off
Measured via `python scripts/arabic_bakeoff.py` on a Colab T4 GPU. Eval set: 2,161 rows — 1,200 MSA (validation split) + 961 Gulf (citizen-grouped 20% holdout carved out of `train`, since `validation` has 0 Gulf rows — see `NOTES.md`).

| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility | Train time |
|---|---:|---:|---:|---:|---:|
| multilingual incumbent (XLM-R) | 1.0000 | 1.0000* | 1.0000 | 1.589 | n/a (reused Lab 3A artefact) |
| CAMeLBERT-mix | 1.0000 | 1.0000 | 1.0000 | 1.305 | 49.0s |
| CAMeLBERT-DA | 1.0000 | 1.0000 | 1.0000 | 1.305 | 53.7s |

\* In-sample/optimistic: the XLM-R incumbent was already fine-tuned on all Gulf rows in Lab 3A, including the ones held out here for the candidates. Not a fair apples-to-apples Gulf comparison for that one cell — see `NOTES.md`.

**Gulf-slice delta vs incumbent: CAMeLBERT-mix +0.00, CAMeLBERT-DA +0.00** macro-F1 points — not the target +4. Same ceiling effect as every other Lab 3/4 classification result on this synthetic corpus: topic vocabulary is trivial enough that all three checkpoints saturate at a perfect 1.0000 regardless of dialect or pretraining. The measurable, non-tied signal here is **AR fertility**: both CAMeLBERT variants tokenize Arabic more efficiently (1.305) than the multilingual XLM-R incumbent (1.589), i.e. shorter, cheaper sequences for the same text — see `DECISIONS.md#arabic-model` for how that (plus Bayan's known dialect mix) breaks the tie between CAMeLBERT-mix and CAMeLBERT-DA.

## Lab 5 — Search
Measured via `python notebooks/05_retrieval_eval.py` on the full 20,000-case FAISS index (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` bi-encoder + `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reranker) against the 150 labelled queries (130 answerable + 20 no-answer), `k=10`, `candidates=50`.

| Configuration | recall@10 (labelled ID) | recall@10 (lenient, text-dup) | topic_precision@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|---:|---:|
| bi-encoder only | 0.0692 | 0.0769 | **1.0000** | 0.0651 | 11.7 ms |
| + cross-encoder rerank | 0.0077 | 0.0692 | **1.0000** | 0.0193 | 222.5 ms |

- no-answer empty-correct: **20 / 20** (min_score = 0.10; a full sweep over 0.02–0.25 gave identical recall and 100% no-answer accuracy at every value tested — the score gap between answerable and no-answer queries is wide, so the exact threshold isn't a knife-edge choice here)
- cross-lingual recall@10 gap (EN − AR, lenient, with rerank): **−0.0262** (AR slightly higher, both very low for the reason below)

**Why recall@10 against the labelled IDs is so low (and why that's not a retrieval failure):** inspected the actual labels — e.g. query Q-001 ("حفرة في الطريق أمام حي العليا منذ 16 أيام") lists `relevant_case_ids` `[CASE-000001, CASE-000009, CASE-000017]`, whose case texts are about damaged asphalt near Jeddah, a road near Al Yasmin, and a road near An Narjis — none textually or situationally close to the query beyond sharing the `roads` topic. Checked the pattern across queries: every query's 3 "relevant" ids are the same topic, spaced by **exactly 8** (Q-001: 1/9/17, Q-002: 10/18/26, Q-003: 19/27/35, ...) — a mechanical, round-robin artifact of how the 8-topic corpus was generated, not a genuine semantic relevance judgement. A content-based retrieval system has no way to preferentially recover an arbitrary same-topic sample like that. The `topic_precision@10 = 1.0000` result (every single one of the top-10 results, for every one of the 130 answerable queries, shares the query's exact topic) is the fairer measure of retrieval quality here, and it's perfect. The corpus is also heavily duplicated (20,000 rows, 5,401 unique `case_text` values, some templates repeated 300+ times), which is why a "lenient" recall counting any textual duplicate of a labelled id as a hit is reported alongside the strict one — it's still low, for the same structural reason.

**Rerank appears to hurt the labelled-ID recall** (0.0692 → 0.0077 strict) while `topic_precision@10` stays perfect either way — expected, not a regression: since the labelled ids are an arbitrary sample within the correct topic, reordering an already-topically-perfect candidate pool by cross-encoder relevance has no reason to preserve that specific arbitrary sample any better than raw bi-encoder similarity order does. No conclusion about rerank quality can be drawn from the labelled-ID metric here; a manual qualitative check (below) is more informative.

**Manual qualitative check** (200-case index, `دفعت الفاتورة` and pothole-style queries): with rerank, an Arabic pothole query correctly retrieved cross-lingual matches including an English "road maintenance" case, and a nonsense/off-topic query (`"كيف الطقس اليوم"`) correctly returned an empty result. The mechanism works as intended; the labelled-ID benchmark just isn't a fair yardstick for it on this fixture.

## Lab 6 — Evaluation
Measured via `python scripts/evaluation_report.py` on `data/eval/validation_predictions.csv` (2,400 rows) plus a live re-inference pass with the actual Lab 3A XLM-R classifier.

| Model | Aggregate accuracy [CI] | Arabic-parks slice | Invariance pass | Directional pass |
|---|---|---:|---:|---:|
| reference fixture (`validation_predictions.csv`) | 0.8750 [0.8617, 0.8888] | 0.0000 | n/a | n/a |
| our XLM-R classifier (re-inferred, same rows) | 1.0000 (parks slice only) | 1.0000 | 0.6000 | 0.8000 |

(Gulf slice: not computable — this fixture's `dialect_region` column only has `MSA`/`NA`, 0 Gulf rows. Full-corpus aggregate accuracy for our re-inferred classifier wasn't separately computed since the paired comparison below is the more informative number.)

- **Paired comparison verdict**: our XLM-R classifier vs the fixture's reference predictions — delta = **+0.1250** [+0.1113, +0.1383] (95% CI entirely positive → our model is significantly better on this set, driven entirely by the parks slice below).
- **Error taxonomy top categories**: 300/300 errors (100%) = new category "Arabic parks↔roads lexical confound" (`docs/ERROR_TAXONOMY.md` #9). Single-category population — see `EVALUATION_REPORT.md` for the full read-through.
- **Top-3 prioritised fixes**: (1) add Arabic `parks` training data not co-located with road vocabulary — predicted delta up to +12.5 accuracy points if the gap is real rather than fixture-specific; (2) add per-`(lang, topic)` slice regression checks to the eval pipeline so a single-slice collapse can't ship unnoticed; (3) verify what model actually produced `validation_predictions.csv` before treating it as current production evidence, since the shipped classifier does not reproduce this failure.
- **Behavioural targets** (course reference: invariance ≥95%, MFT/directional ≥90%): **not met** — invariance 60.0%, directional 80.0%, measured on the real classifier, reported honestly rather than adjusted.

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
