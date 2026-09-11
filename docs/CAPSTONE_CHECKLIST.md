# Final Capstone — Bayan Service Checklist

> Checked where measured evidence supports it. Left unchecked (with a one-line reason and a pointer to the evidence) where the synthetic data made a target structurally unreachable, rather than silently claiming it. See `NOTES.md`/`BENCHMARKS.md` for full detail on every item below.

## Mandatory
- [x] One versioned bilingual + Arabic-profile preprocessing contract used by train/eval/serve. (`preprocess()`/`PREPROC_VERSION`, checked at serve time by `canaries.py`.)
- [x] PII recall 100% on the 60-case fixture. (Lab 1.)
- [x] Segmentation consistent train-to-serve; startup skew canaries green. (`/health` reports `canaries_ok: true`; Lab 4's `segment()` was evaluated for NER LOCATION recall specifically, not wired into the main classifier path — see NOTES.md.)
- [ ] Topic classifier beats TF-IDF baseline by ≥8 macro-F1. **Not reachable**: the TF-IDF baseline itself scores a perfect 1.0000 macro-F1 (heavily templated synthetic vocabulary — see BENCHMARKS.md Lab 3), so no classifier can mathematically beat it by 8 points. The XLM-R classifier ties the baseline at 1.0000 after the 3-epoch GPU retrain.
- [ ] Dialect-aware variant has a CI-backed Gulf-slice improvement. **Not reachable** for the same ceiling-effect reason: all three Lab 4 bake-off checkpoints (XLM-R, CAMeLBERT-mix, CAMeLBERT-DA) tie at 1.0000 on a genuine held-out Gulf slice. CAMeLBERT-DA was still chosen, on AR-fertility + traffic-mix evidence — see `DECISIONS.md#arabic-model`.
- [x] NER entity-F1 ≥0.80; clitic/alignment tests green. (1.0000; 8/8 alignment tests.)
- [x] Extractive QA has honest null handling; required no-answer target met. (9/9 answerable, 3/3 null, zero-shot.)
- [x] Versioned FAISS index + manifest + two-stage reranking. (Lab 5.)
- [ ] recall@10 ≥0.80 and MRR@10 ≥0.70; cross-lingual gap reported. **Not reachable as literally stated**: the supplied "relevant" case IDs are a mechanical same-topic sample (spaced exactly 8 apart), not genuine relevance judgements — recall@10 against them is ~0.07 regardless of retrieval quality. `topic_precision@10 = 1.0000` (every top-10 result, every query, correctly on-topic) is reported as the fairer measure — see NOTES.md/BENCHMARKS.md Lab 5. Cross-lingual gap on the lenient metric: −0.0262.
- [x] Sliced evaluation: language/dialect/class/length + bootstrap CIs. (Lab 6.)
- [ ] Behavioural suite ≥95% invariance and ≥90% MFT. **Not met**: measured invariance 60.0%, directional (MFT proxy — no sentiment model in this project) 80.0%, on the real deployed classifier. A genuine, currently-unresolved robustness gap, reported rather than hidden behind the near-perfect aggregate accuracy.
- [x] ≥100 hand-read errors in final report (Lab 6 works with 120) + top-3 prioritised fixes. (300 errors read; population turned out 100% homogeneous — one root cause, three fix angles, documented in `EVALUATION_REPORT.md`.)
- [x] One model card per artefact with hand-written limitations. (3 cards in `docs/model_cards/`.)
- [ ] Classifier HTTP p99 ≤40 ms at 16 concurrent on lab CPU. **Not met**: best measured configuration is 63.41 ms (down from 113.94 ms with no thread pinning). Investigated as a genuine single-machine CPU concurrency limit for a 270M-parameter model, not a config bug — three thread configurations were tried and measured; see `BENCHMARKS.md` Lab 7 HTTP load test.
- [x] Full optimisation ladder + paired quality taxes; fp32 rollback retained. (38.6x/28.1x speed-up, 0.0000 quality tax, fp32 kept for canary comparison.)
- [x] `DECISIONS.md` explains model-family/checkpoint choices with fertility/slice evidence.
- [x] Re-runnable scripts; frozen test untouched until final report; participant-owned benchmark numbers.
- [x] Meaningful commit history across the project's build sessions.

## Chosen extension
- [x] **Batch endpoint `/v1/classify:batch`** — accepts `{"texts": [...]}`, returns per-text classification, reusing the same INT8 serving path as `/v1/classify`.

## Final deliverables
- [x] Repository with full commit history
- [x] `make serve` works from a clean clone and startup canaries are green (verified live: `/health` → `canaries_ok: true`, `search_ready: true`)
- [x] `EVALUATION_REPORT.md`
- [x] `BENCHMARKS.md`
- [x] `DECISIONS.md`
- [ ] 5-minute bilingual live demo — a live action for the presenter (classification + entities + similar cases; explain one behavioural test; defend one metric claim), not a repository artifact. All the pieces it would use are wired and verified working (`/v1/analyse` demonstrated live against real Arabic and English text during capstone integration — see NOTES.md).
