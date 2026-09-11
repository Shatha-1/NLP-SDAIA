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
| Lab 6 — Evaluation Report | ⬜ Not started |
| Lab 7 — Optimisation & Serving | ⬜ Not started |
| Capstone | ⬜ Not started |

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

---

## 🧭 Evidence Files

- **`NOTES.md`** — observations, defect analysis, and findings from every lab.
- **`BENCHMARKS.md`** — numbers from actual runs, never copied reference values.
- **`DECISIONS.md`** — model/tokenizer/architecture choices backed by measured evidence.
