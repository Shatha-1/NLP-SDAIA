# Bayan
## SDA-AIE-211 — Natural Language Processing with Transformers

> **From raw bilingual text to a working NLP service.**
>
> Bayan is a bilingual (Arabic/English) citizen-feedback intelligence service, built lab by lab across this course: preprocessing, transformer fundamentals, fine-tuned topic classification, NER, extractive QA, Arabic dialect handling, semantic search, evaluation, and an optimised, served pipeline.

---

## 👤 Author

**Shatha Hamad Bin Mana**
[SDAIA Academy](https://github.com/SDAIAAcademy)

---

## ✅ Progress

| Lab | Status |
|---|---|
| Lab 1 — Bilingual Preprocessing & Tokenisation | ✅ Complete |
| Lab 2 — Transformer Attention | ✅ Complete |
| Lab 3 — Topic Classification, NER, Extractive QA | ✅ Complete |
| Lab 4 — Arabic Pipeline & Dialect-Aware Fine-tuning | ✅ Complete |
| Lab 5 — Bilingual Semantic Search | ✅ Complete |
| Lab 6 — Evaluation Report | ✅ Complete |
| Lab 7 — Optimisation & Serving | ✅ Complete |
| Capstone | ✅ Complete |

Full measured evidence lives in [`BENCHMARKS.md`](BENCHMARKS.md), engineering notes and findings in [`NOTES.md`](NOTES.md), and evidence-backed decisions in [`DECISIONS.md`](DECISIONS.md).

---

## 🏗️ Pipeline

```text
Raw Citizen Feedback
        ↓
Versioned Preprocessing
        ↓
Topic Classification / NER / QA
        ↓
Arabic-aware Model Decisions
        ↓
Semantic Search → FAISS → Re-ranking
        ↓
Evaluation + Model Cards + Benchmarks
        ↓
ONNX / INT8 Optimisation
        ↓
FastAPI Bayan Service
```

---

## 📁 Repository Structure

```text
SDA-AIE-211-Bayan/
│
├── src/bayan/
│   ├── preprocessing/
│   │   ├── core.py             # Lab 1 — normalize / mask_pii / preprocess
│   │   ├── segmentation.py     # Lab 1 — sentence segmentation
│   │   └── arabic.py           # Lab 4 — Arabic normalisation + clitic segmentation
│   ├── attention.py            # Lab 2 — scaled dot-product attention + MHA
│   ├── models/
│   │   ├── data.py             # Lab 3A — grouped dataset loading
│   │   ├── ner.py              # Lab 3B — BIO label alignment
│   │   └── qa.py               # Lab 3B — constrained span search
│   ├── search/                 # Lab 5
│   ├── evaluation/              # Lab 6
│   └── serving/                 # Lab 7 + Capstone
│
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_tokenizer_audit.py   # Lab 1
│   └── 02_transformer_anatomy.py # Lab 2
│
├── scripts/
│   ├── doctor.py
│   ├── parameter_audit.py      # Lab 2
│   ├── tfidf_baseline.py       # Lab 3A
│   ├── train_classifier.py     # Lab 3A
│   ├── train_ner.py            # Lab 3B
│   ├── qa_smoke.py             # Lab 3B
│   ├── dialect_audit.py        # Lab 4
│   └── arabic_bakeoff.py       # Lab 4
│
├── tests/
├── data/
├── artifacts/                  # trained model artefacts (gitignored)
│
├── NOTES.md
├── BENCHMARKS.md
├── DECISIONS.md
├── requirements.txt
├── pyproject.toml
└── Makefile
```

---

## ⚙️ Setup

Targets Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/doctor.py
```

For GPU-heavy training (Labs 3–4), use `notebooks/00_colab_setup.ipynb` to clone this same repository into a Colab GPU runtime.

---

## 📊 What's been built

### Lab 1 — Bilingual Preprocessing & Tokeniser Decision

Built the shared `normalize()` / `mask_pii()` / `preprocess()` contract used by every later lab: Unicode NFKC normalisation, tatweel and HTML-tag stripping, repeated-character collapsing (3+ repeats → 2), whitespace cleanup, and regex-based masking of Saudi phone numbers and national-ID-shaped values. A rule-based spaCy sentence segmenter runs on top of it.

- **25/25** golden preprocessing pairs pass; **60/60 (100%)** PII masking recall.
- Audited 4 tokenizer candidates (mBERT, XLM-R, CAMeLBERT, DistilBERT) on 12,000 real feedback rows for AR/EN fertility, p95 sequence length, and UNK rate.
- **Decision: XLM-R** — the best balanced bilingual fertility (1.672 AR / 1.434 EN) with a 0.00% Arabic UNK rate, chosen over CAMeLBERT (Arabic-only) and DistilBERT (English-only) since Bayan's real traffic code-switches between languages in the same message. Full rationale in `DECISIONS.md`.

### Lab 2 — Transformer Attention From Scratch

Implemented scaled dot-product attention and multi-head attention in raw PyTorch, verified bit-for-bit against `F.scaled_dot_product_attention` (atol=1e-6), and built a causal mask that produces a provably lower-triangular attention matrix (decoder-style, GPT-family behaviour).

- Audited parameter distribution for mBERT vs CAMeLBERT — mBERT spends 51.8% of its 177.9M parameters on embeddings (119,547-token multilingual vocabulary) vs CAMeLBERT's 21.5% (30,000-token Arabic-only vocabulary), quantifying the "multilingual tax."
- Diagnosed a real pad-attention leak on live Bayan text: with `attention_mask` passed, mean attention mass on `[PAD]` tokens is exactly 0; without it, real tokens leak ~2.7% of their attention budget onto padding — silent degradation with no crash, which is why the Lab 7 serving path must always pass the mask.

### Lab 3 — Topic Classification, NER, Extractive QA

**3A — Topic classifier:** grouped train/validation/test split (0 citizen overlap, verified), a TF-IDF + LinearSVC baseline, and a fine-tuned XLM-R classifier. First run (1 epoch, CPU) left a real 20-point validation/test gap (1.0000 vs 0.7993); retraining for 3 epochs on a Colab T4 GPU closed it completely — **frozen test macro-F1 = 1.0000**, matching the baseline.

**3B — NER:** implemented BIO-to-subword label alignment (`-100` masking for continuation pieces and special tokens, 8/8 contract tests), then fine-tuned token classification on 4,000 CoNLL sentences — **entity-F1 = 1.0000** across DATE/LOCATION/REFERENCE/SERVICE, well above the ≥0.80 target.

**3B — QA:** implemented constrained span search with honest SQuAD2-style null handling (rejects inverted spans, respects a null-score threshold), then ran it zero-shot against `deepset/roberta-base-squad2` on a self-built 9-answerable + 3-unanswerable smoke set — **9/9 correct spans, 3/3 correct nulls**.

Every ceiling-effect result (e.g. a perfect baseline leaving no room to "beat by +0.08") is measured and explained in `BENCHMARKS.md`, not smoothed over.

### Lab 4 — Arabic Pipeline & Dialect-Aware Fine-tuning

- `normalize_arabic()`: hamza normalisation, teh marbuta → heh, alef maksura → yeh, tatweel removal, optional dediacritisation — **30/30** golden pairs pass.
- `segment()`: CAMeL Tools MLE-disambiguator clitic segmentation (D3 scheme), with a fallback for colloquial words the morphology database can't analyse.
- Dialect audit: Arabic traffic is **66.7% Gulf / 33.3% MSA** — evaluating only on MSA would silently miss most real Arabic feedback.
- NER + segmentation re-evaluation: LOCATION recall was already a perfect 1.0000 pre-segmentation, so the measured delta is 0.00 (not +4) — a genuine ceiling effect, documented rather than hidden.
- **Arabic model bake-off** (CAMeLBERT-mix vs CAMeLBERT-DA vs the XLM-R incumbent): discovered the frozen test split is 100% English and `validation` has zero Gulf rows, so a genuine held-out Gulf slice (961 rows) was carved out of `train` by citizen group before evaluating. All three checkpoints tied at a perfect 1.0000 macro-F1 on every slice (another ceiling effect — Gulf-slice delta vs incumbent: +0.00 for both candidates). With accuracy tied, the tie-break used AR fertility (both CAMeLBERT variants tokenize Arabic 18% more efficiently than XLM-R) plus Bayan's known Gulf-majority traffic — **decision: CAMeLBERT-DA**, full rationale in `DECISIONS.md`.

### Lab 5 — Bilingual Semantic Search

Built a two-stage search service over 20,000 historical cases: a bi-encoder (`paraphrase-multilingual-MiniLM-L12-v2`) + FAISS `IndexFlatIP` for retrieval, a cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) for reranking, and a versioned manifest (model, preprocessing version, vector count, dimension) that the service asserts on load.

- Found and fixed a real bug: the cross-encoder returns unbounded raw logits, not a [0,1] score — the initial `min_score=0.25` threshold silently rejected every query, including relevant ones. Fixed with a sigmoid transform, then tuned `min_score` via a sweep against the 20 labelled no-answer queries — **20/20 correctly rejected**, robust across the whole tested range.
- Investigated a suspiciously low recall@10 (~0.07) rather than assuming the target was unreachable: the supplied "relevant" case IDs turned out to be a mechanical same-topic sample spaced exactly 8 apart (a corpus-generation artifact), not genuine relevance judgements — confirmed by inspecting actual query/answer text pairs. Added `topic_precision@10` as a fairer diagnostic: **1.0000** — every single top-10 result for every query was correctly on-topic, proving the retrieval mechanism itself works even though the labelled-ID recall metric doesn't fairly measure it here.
- Full reasoning and a manual qualitative cross-lingual retrieval check are in `NOTES.md`.

### Lab 6 — Evaluation Report

Built bootstrap confidence intervals (`bootstrap_ci`, `paired_bootstrap_diff`), a sliced-accuracy report with small-slice flagging, a behavioural test runner (invariance/directional), and an automated evaluation-report + model-card generator (3 model cards in `docs/model_cards/`, known-limitations sections written by hand per the lab's requirement).

- Sliced the 2,400-row validation fixture and found a single, complete blind spot: every Arabic `parks` case is misclassified as `roads` (0/300 correct) while every other slice scores a perfect 1.0000 — investigated instead of just reporting the aggregate 87.5% accuracy.
- Re-ran the actual Lab 3A XLM-R classifier on those exact same 300 rows: **1.0000** — the blind spot lives in the reference prediction fixture, not in the model this project trained. Confirmed statistically with a paired bootstrap (delta +12.5 points, 95% CI entirely positive) and documented as a new error-taxonomy category (`docs/ERROR_TAXONOMY.md` #9) rather than assumed away.
- Behavioural suite (adapted to use the real topic classifier, since no sentiment model exists in this project — see `NOTES.md`): invariance 60.0%, directional 80.0%, both below the course's reference targets — a genuine, unresolved robustness gap reported honestly rather than hidden behind the strong aggregate accuracy.
- Full write-up in `EVALUATION_REPORT.md`.

### Lab 7 — Optimisation & Serving

Built the benchmark ladder (fp32 torch → ONNX fp32 → ONNX INT8), a FastAPI service with startup canaries, and a load test.

- **38.6x p50 / 28.1x p99 speed-up** from fp32 torch @512-padded (302ms/351ms) to ONNX INT8 @128 (7.83ms/12.51ms), with **zero measured quality tax** on both the topic classifier and NER (paired bootstrap, macro-F1/entity-F1 unchanged) — INT8 ships for both models, fp32 kept as the rollback artefact.
- Caught a noisy single-run result before trusting it: an n=200 benchmark showed INT8 with a *worse* p99 than fp32-ONNX; re-running at n=500 with more warm-up showed this was sampling noise (INT8 is actually best on both p50 and p99).
- **HTTP load test never meets the 40ms p99 target** (16 concurrent, 60s): started at 113.94ms with no thread pinning, cut to 63.41ms by pinning `OMP_NUM_THREADS`/ONNX `intra_op_num_threads=4`, and got *worse* (132.26ms) trying `intra_op_num_threads=1`. Concluded this is a genuine single-machine CPU concurrency limit for a 270M-parameter model, not something more thread-tuning fixes — reported the real number rather than only showing the single-request latency that does meet target.
- `hey` wasn't available in this environment (no Go toolchain either); wrote `scripts/load_test.py`, a `ThreadPoolExecutor`-based equivalent with the same fixed-concurrency/fixed-duration spec, rather than downloading an untrusted prebuilt binary.
- Startup canaries (`canaries.py`) compare the serving (INT8) artefact's prediction against the fp32 rollback on a pinned input, failing fast if a quantised export is corrupted; `/health` reports canary status.

### Capstone — Assembling Bayan

Wired the remaining endpoints on top of the Lab 7 service: `POST /v1/entities` (NER), `POST /v1/search` (Lab 5 two-stage search), `POST /v1/analyse` (classification + entities + similar cases in one call), plus a chosen extension, `POST /v1/classify:batch`. Full test suite: **78/78 passing**.

- Live-tested every endpoint against the running service (not just the pytest contract) with real bilingual text — classification and search performed excellently; NER on a sentence phrased differently from its narrow training templates missed part of a location span, which is exactly the "unvalidated on novel phrasing" limitation already written by hand into the NER model card — now empirically demonstrated rather than just hypothesised.
- Caught a testing artefact while doing this: Windows console encoding mangled Arabic text passed as an inline `curl -d "..."` argument, briefly looking like a server bug (missing entities, empty search results) until isolated to the test harness, not the API, by calling the code directly in Python and by switching to a UTF-8 file payload.
- `docs/CAPSTONE_CHECKLIST.md` is filled in honestly: several course-reference targets (classifier beating an already-perfect TF-IDF baseline, retrieval recall@10 against a mechanically-sampled label set, HTTP p99 ≤40ms on a single CPU) are marked **not met** with the measured evidence and root-cause explanation, rather than adjusted to look passing — consistent with every other ceiling-effect and data-quirk finding across this project.

---

## 🧭 Evidence Files

- **`NOTES.md`** — observations, defect analysis, and findings from every lab.
- **`BENCHMARKS.md`** — numbers from actual runs, never copied reference values.
- **`DECISIONS.md`** — model/tokenizer/architecture choices backed by measured evidence.
- **`EVALUATION_REPORT.md`** — the Lab 6 honest quality report: sliced metrics, behavioural rates, error taxonomy, known limitations.
- **`docs/model_cards/`** — per-model cards (intended use, metrics, known limitations).
