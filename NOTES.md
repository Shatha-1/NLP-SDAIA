# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1
- Class: PII (phone numbers, Saudi national ID / iqama numbers)
- Example: `FB-000001`: "...عاجلة 😡 <br> 0551234567 1023456789" contains both a `05XXXXXXXX` mobile number and a 10-digit national-ID-shaped number.
- Why it matters: raw PII must never reach model training/inference or logs; leaking it is a compliance/privacy risk.
- Decision: clean (mask) — replaced with `<PHONE>` / `<NATIONAL_ID>` in `mask_pii()`.

### Defect 2
- Class: Emoji
- Example: `FB-000027`: "...ولم تُفرغ منذ 6 أيام 😡"
- Why it matters: emoji carry real sentiment signal (frustration, urgency) for the topic/sentiment classifiers in later labs.
- Decision: preserve — `normalize()` leaves emoji untouched.

### Defect 3
- Class: HTML remnants
- Example: `FB-000001`, `FB-000032`, `FB-000063`, `FB-000094`: literal `<br>` tags left in scraped/pasted feedback text.
- Why it matters: markup fragments are not linguistic content and would be tokenized as noisy garbage subwords, hurting fertility and model quality.
- Decision: clean — strip HTML tags before/within preprocessing (not linguistic signal).

### Defect 4
- Class: Whitespace irregularities (leading/trailing spaces, doubled spaces, tabs/newlines)
- Example: `FB-000008`: "  ألعاب الأطفال في حديقة حي العليا تحتاج صيانة   " (leading/trailing spaces); `FB-000001` has multiple internal double-spaces after `<br>`/PII removal.
- Why it matters: inconsistent whitespace inflates token counts inconsistently and breaks exact-match evaluation/golden tests.
- Decision: clean — `normalize()` collapses runs of whitespace to a single space and strips ends.

### Defect 5
- Class: Character elongation (informal repeated letters for emphasis, tatweel-style stretching)
- Example: `FB-000088`, `FB-000117`, `FB-000146`: "لووووسمحت" (و stretched 4x for "please").
- Why it matters: elongation is orthographic noise that fragments the same word into many surface forms, hurting the tokenizer's fertility and the classifier's ability to generalize; but collapsing too aggressively can erase true double letters ("ممتاز" vs "ممتاااز" both mean "excellent"), so we cap at 2 repeats instead of 1.
- Decision: clean, but conservatively — `normalize()` collapses 3+ repeats of the same character down to 2 (not 1), matching the course golden contract.

### Defect 6
- Class: Code-switching / mixed-script reference codes
- Example: `FB-000004`: "There is a water leak at Jeddah; reference BYN-2026-000003" and Arabic rows like `FB-000013`: "...على الفاتورة رقم BYN-2025-000012" mix Arabic sentences with Latin-alphanumeric case IDs.
- Why it matters: these alphanumeric codes are high-value entities (useful for NER/lookup in Lab 3B/5) but would inflate fertility for word-piece tokenizers and should not be masked like PII since they are not personal data.
- Decision: task-dependent — preserve as-is in preprocessing (not PII, not noise); flagged for the NER lab as a candidate entity type.

### Bonus defects observed (beyond the required 6)
- **ALL-CAPS shouting** (`FB-000020`: "THERE IS A WATER LEAK AT DAMMAM; REFERENCE BYN-2026-000019") — preserved; case can carry emphasis/sentiment and normalization does not lowercase text.
- **Informal hamza-alef spelling variants** (`FB-000001` "ألطريق" vs. standard "الطريق"; `FB-000058` "ألإنارة") — left untouched in Lab 1's `normalize()`; this is an Arabic-specific orthographic normalization deferred to Lab 4 (`normalize_arabic()`), not the shared bilingual contract.

## Lab 1 — Sentence segmentation spot-check
Pipeline: `spacy.blank("xx")` + `sentencizer` (rule-based, language-agnostic — no model download needed since AR and EN share the same pipeline object) applied on top of `preprocess()`.

The 5 longest examples in `bayan_raw_sample.csv` (FB-000079, FB-000001, FB-000014, FB-000170, FB-000135) are all single-sentence complaints with no `.`/`؟` punctuation — none contains a numbered-list complaint, so that case was spot-checked with a synthetic multi-sentence example instead. Findings:

- **Single run-on complaints are correctly kept as one sentence** — the pipeline does not spuriously split on an em dash (`—`) inside "...Maintenance Appointments — very frustrating 😡", which is the right behaviour (it's one clause, not two sentences).
- **Real sentence boundaries split correctly** — `"There is a water leak... reference BYN-2026-000003. Please investigate."` splits into two sentences at the period, without breaking the hyphenated `BYN-2026-000003` reference code itself.
- **Noteworthy side effect found while spot-checking**: because `split_sentences()` runs `preprocess()` first, `normalize()`'s repeated-character collapse (3+ repeats → 2) also collapses runs of repeated **digits**, e.g. `BYN-2026-000003` → `BYN-2026-003` (five zeros collapsed to two). This is harmless for sentiment/topic classification but would corrupt an exact case-ID lookup. Documented as a known limitation: **downstream features that need the exact case ID (Lab 5 search, Lab 3B NER) should read the ID from the raw/original text, not the normalized text.**
- Duplicate-word feedback ("My My Licence licence request has been under review for 15 days") is preserved as one sentence and the duplicate words are not merged — consistent with the Lab 1 decision to treat repeated *words* as signal, unlike repeated *characters* within a word.

## Lab 2 — Parameter audit
Measured via `python scripts/parameter_audit.py`.

| Checkpoint | Total params | Embeddings % | Attention % | FFN % | Other notes |
|---|---:|---:|---:|---:|---|
| mBERT | 177,853,440 | 51.8% | 15.9% | 31.9% | vocab_size = 119,547 |
| CAMeLBERT | 109,081,344 | 21.5% | 26.0% | 52.0% | vocab_size = 30,000 |

**Why is the embedding share different?** Embedding params scale directly with `vocab_size × hidden_size`; mBERT's vocabulary (119,547 subwords, covering ~100+ languages) is ~4x larger than CAMeLBERT's Arabic-focused vocabulary (30,000 subwords), so mBERT pays a much bigger "multilingual tax" — over half its parameters just store the embedding table, leaving proportionally less capacity for the actual transformer layers (attention+FFN), even though both share the identical 12-layer/768-hidden encoder architecture.

## Lab 2 — Causal mask
Verified in `notebooks/02_transformer_anatomy.py`: with a lower-triangular mask, the resulting attention weight matrix is confirmed lower-triangular (`torch.allclose(w, torch.tril(w))` → True). This is **decoder-style causal (autoregressive) attention**, the pattern used in GPT-family models where position *i* may only attend to positions ≤ *i*.

## Lab 2 — Attention-map diagnostics + pad leak
Real Bayan example tokenized with mBERT: `['[CLS]', 'ال', '##ح', '##اوية', ..., '[SEP]']` (16 real tokens + padding to match batch).

- **[CLS] "sink" heads**: heads 7 and 8 (last layer) put unusually high weight on `[CLS]` attending to itself (0.480, 0.542) — a common pattern where a head effectively "parks" on a fixed anchor token rather than aggregating new information.
- **[SEP] sink**: not a uniform pattern across all heads (mean [CLS]→[SEP] weight = 0.176, below a 0.3 threshold), but two individual heads (4 and 11) show a stronger pull toward `[SEP]` (0.220, 0.336) — so the "attention sink" behaviour documented in the literature shows up head-by-head here, not as a whole-layer effect. Do not eyeball a single head and generalise to "the model" — check the full head-by-head table.
- **Adjacency-looking head**: head 6 puts its top [CLS] weight on `ال` — the token immediately following [CLS] (position 1) — consistent with a local/adjacent-token attention pattern.
- **Pad-leak diagnosis (the planted bug)**: running the same batch through mBERT *with* `attention_mask` gives **0.00000** mean attention mass on `[PAD]` positions (correctly zeroed by the mask). Running it *without* `attention_mask` gives **0.02694** — real tokens leak ~2.7% of their attention budget onto meaningless `[PAD]` positions. This directly explains why Lab 7's serving path must always pass `attention_mask`: skipping it doesn't crash anything, it just silently degrades every prediction on a padded batch.

## Lab 3A — Grouped split
`build_topic_dataset()` loads `data/raw/bayan_feedback.csv` and partitions by its existing `split` column rather than re-shuffling with a fresh grouped split. `data/DATA_DICTIONARY.md` states the supplied split is "70/20/10 and deterministic. The final test split is frozen" and already has 0 `citizen_group_id` overlap across train/validation/test (verified directly) — rebuilding it would risk breaking that frozen contract, so this loads and filters instead of re-grouping.

## Lab 3A — TF-IDF baseline vs fine-tuned classifier
The TF-IDF + LinearSVC baseline scored a perfect **1.0000 macro-F1** on both validation and the frozen test split — see the note under `BENCHMARKS.md`'s Lab 3 table (the dataset only has 3,820 unique underlying texts recycled across 12,000 rows, each topic with near-exclusive vocabulary, so bag-of-words alone solves it).

The fine-tuned XLM-R classifier (1 epoch, CPU) matched validation (1.0000) but scored **0.7993** on the frozen test split — a real 20-point val/test gap, most likely because near-duplicate template sentences let the model partly memorize its way to a perfect validation score after only one epoch, while the frozen test split's mix generalised less well from that single pass. Documented as measured evidence, not smoothed over — see `BENCHMARKS.md` for the full explanation. A follow-up worth trying (not required by the lab target): more epochs or a lower learning rate to see whether the gap closes, and de-duplicating near-identical templates across splits.

## Lab 3B — NER training
Trained XLM-R token classification on `data/models/bayan_ner.conll` (4,000 sentences, 80/10/10 split by sentence, 1 epoch). Reached a perfect **1.0000 entity-F1** on validation and frozen test for all 4 present entity types (DATE, LOCATION, REFERENCE, SERVICE) — comfortably clears the ≥0.80 target. `ORGANISATION` (listed in `DATA_DICTIONARY.md`'s NER schema) does not actually appear anywhere in `bayan_ner.conll` (verified: only `O`, `B-DATE`, `B-LOCATION`, `B-REFERENCE`, `B-SERVICE` exist in the file), and there are no `I-` continuation tags either — every entity in this dataset is a single BIO-tagged "word" chunk (which may itself contain an internal space, e.g. `"خدمات المياه"` tagged as one `B-SERVICE` unit). Verified that `is_split_into_words=True` correctly assigns the same `word_id` to both subword pieces of such a multi-space chunk before relying on it for training.

## Lab 3B — QA smoke set
**Data quirk found**: the supplied `data/eval/qa_smoke_set.json` (12 questions) has `is_impossible=false` for all 12 rows — it cannot supply the "3 unanswerable" half of the lab's 9/3 target on its own, and its 12 rows are really just 3 unique questions about "Service 01" repeated 4x each (not diverse). The actual unanswerable examples exist in the larger training fixture `data/models/bayan_qa.json` (600 QAs total, 100 `is_impossible=true`). `scripts/qa_smoke.py` therefore builds its own 12-question smoke set directly from `bayan_qa.json` — 9 unique answerable questions (3 each from Service 01/02/03) + 3 unique unanswerable questions — instead of trusting the incomplete supplied fixture.

Lab 3B has no QA fine-tuning script (unlike the classifier and NER labs), so `best_span()` is exercised **zero-shot** against `deepset/roberta-base-squad2`, an off-the-shelf SQuAD2 checkpoint already trained with null-answer support. Result: **9/9 answerable → correct span, 3/3 unanswerable → answer=None**, meeting the lab's target with no Bayan-specific training at all — a reasonable outcome since SQuAD2 pretraining already covers exactly this "extract span or abstain" skill on English text.

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
