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

The fine-tuned XLM-R classifier (1 epoch, CPU) matched validation (1.0000) but scored **0.7993** on the frozen test split — a real 20-point val/test gap, most likely because near-duplicate template sentences let the model partly memorize its way to a perfect validation score after only one epoch, while the frozen test split's mix generalised less well from that single pass. Documented as measured evidence, not smoothed over — see `BENCHMARKS.md` for the full explanation.

**Follow-up (confirmed the theory):** retrained for 3 epochs on a Colab T4 GPU (same code/data/split) — frozen test macro-F1 went from 0.7993 to a perfect **1.0000**, and training loss kept dropping steadily each epoch (0.4537 → 0.0105 → 0.0043). This confirms the 1-epoch CPU run was simply undertrained, not evidence of a deeper generalisation problem.

**Important correction found while building Lab 4's bake-off**: the frozen test split is **100% English** — `ds["test"]["lang"].value_counts()` is `{"en": 1200}`, zero Arabic rows. Full per-split language breakdown:

| Split | ar | en |
|---|---:|---:|
| train | 6,000 | 2,400 |
| validation | 1,200 | 1,200 |
| test | 0 | 1,200 |

This means every "frozen test macro-F1" number reported above (TF-IDF baseline, both classifier runs) was measured **on English text only**, not bilingually as assumed when those results were first written up. It doesn't invalidate the measurements — the classifier genuinely does score 1.0000 macro-F1 on that English test slice — but the claim should be read as "English topic classification," not "bilingual topic classification." `validation` is the only split with a balanced AR/EN mix (and isn't used for any tuning decision in this project, so it's safe to reuse for language-sliced evaluation); Lab 4's Arabic model bake-off uses `validation` instead of `test` for exactly this reason, documented inline in `arabic_bakeoff.py`.

## Lab 3B — NER training
Trained XLM-R token classification on `data/models/bayan_ner.conll` (4,000 sentences, 80/10/10 split by sentence, 1 epoch). Reached a perfect **1.0000 entity-F1** on validation and frozen test for all 4 present entity types (DATE, LOCATION, REFERENCE, SERVICE) — comfortably clears the ≥0.80 target. `ORGANISATION` (listed in `DATA_DICTIONARY.md`'s NER schema) does not actually appear anywhere in `bayan_ner.conll` (verified: only `O`, `B-DATE`, `B-LOCATION`, `B-REFERENCE`, `B-SERVICE` exist in the file), and there are no `I-` continuation tags either — every entity in this dataset is a single BIO-tagged "word" chunk (which may itself contain an internal space, e.g. `"خدمات المياه"` tagged as one `B-SERVICE` unit). Verified that `is_split_into_words=True` correctly assigns the same `word_id` to both subword pieces of such a multi-space chunk before relying on it for training.

## Lab 3B — QA smoke set
**Data quirk found**: the supplied `data/eval/qa_smoke_set.json` (12 questions) has `is_impossible=false` for all 12 rows — it cannot supply the "3 unanswerable" half of the lab's 9/3 target on its own, and its 12 rows are really just 3 unique questions about "Service 01" repeated 4x each (not diverse). The actual unanswerable examples exist in the larger training fixture `data/models/bayan_qa.json` (600 QAs total, 100 `is_impossible=true`). `scripts/qa_smoke.py` therefore builds its own 12-question smoke set directly from `bayan_qa.json` — 9 unique answerable questions (3 each from Service 01/02/03) + 3 unique unanswerable questions — instead of trusting the incomplete supplied fixture.

Lab 3B has no QA fine-tuning script (unlike the classifier and NER labs), so `best_span()` is exercised **zero-shot** against `deepset/roberta-base-squad2`, an off-the-shelf SQuAD2 checkpoint already trained with null-answer support. Result: **9/9 answerable → correct span, 3/3 unanswerable → answer=None**, meeting the lab's target with no Bayan-specific training at all — a reasonable outcome since SQuAD2 pretraining already covers exactly this "extract span or abstain" skill on English text.

## Lab 4 — Arabic normalisation (golden contract)
`normalize_arabic()` always applies: NFKC, tatweel removal, hamza-on-alef → bare alef (آ/أ/إ/ٱ → ا), alef maksura → yeh (ى → ي), teh marbuta → heh (ة → ه), hamza-on-waw/yeh → bare waw/yeh. Diacritics are stripped only when `profile.dediacritize=True`, so an untested "display" profile (`dediacritize=False`) can keep them for showing raw text to users while a "model" profile gets the fully flattened form. 30/30 golden pairs pass.

**Data quirk**: `data/eval/arabic_normalize_golden.csv` only contains one profile name (`bayan_ar_v1`, repeated 3x per pair = 30 rows from 10 unique pairs) — the "two course profiles" the README describes aren't both represented in the fixture, similar to the Lab 1 raw-sample and Lab 3B smoke-set quirks. The dediacritize=False path is implemented per the course description but isn't exercised by any golden test.

## Lab 4 — Clitic segmentation
`segment()` uses CAMeL Tools' MLE disambiguator D3 scheme (`d3seg`), splitting both proclitics (e.g. `ال+`) and enclitics (e.g. `+ه`) off their stems, with dediacritization per piece. Verified `"مرجعه"` → `["مرجع", "ه"]`, matching the course-supplied `bayan_ner_segmented.conll` reference exactly for that word. Added a fallback for CAMeL Tools' `"NOAN"` (no-analysis) marker — informal/colloquial words like `"لووووسمحت"` aren't in the MSA morphology DB and returned the literal string `"NOAN"` before the fix; now they fall back to the raw word unchanged.

**Data quirk found**: my D3-based `segment()` is *more* fine-grained than the course-supplied `bayan_ner_segmented.conll` — it also splits the definite article `ال+` off nouns (e.g. `"الدمام"` → `["ال", "دمام"]`), while the reference file leaves `"الدمام"` and `"خدمات المياه"` completely unsplit and only splits the enclitic pronoun in `"مرجعه"` → `"مرجع"`+`"ه"`. Since re-deriving my own split would also require re-deriving new BIO alignments (the reference file's tags are already aligned to its own lighter split), the NER re-training/recall comparison below uses the **course-supplied `bayan_ner_segmented.conll` directly** rather than regenerating it from `segment()` — `segment()` itself is implemented, tested, and correct as a standalone CAMeL Tools D3 utility, it's just a different (more aggressive) scheme than the specific fixture the recall-delta comparison was built against.

## Lab 4 — Dialect audit
Measured via `python scripts/dialect_audit.py` on `data/raw/bayan_feedback.csv`.

- **Distribution** (Arabic rows only, n=7,200): Gulf 4,800 (66.7%), MSA 2,400 (33.3%). (English rows: `dialect_region="NA"`, not applicable.)
- **One-sentence implication for MSA-only evaluation**: two-thirds of real Arabic feedback is Gulf dialect, so evaluating (or fine-tuning) only on MSA text would silently miss quality problems for the majority of actual Arabic traffic.

## Lab 5 — Search index + service contract
`build_index()` encodes `case_text` with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, L2-normalises, and builds a `faiss.IndexFlatIP` (inner product on normalised vectors = cosine similarity). Manifest pins `model`, `preproc_version`, `n_vectors`, `dim`; `CaseSearch.__init__` asserts the index/metadata/manifest agree before serving. Contract test passes.

**Cross-encoder score scale bug (found and fixed)**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` returns raw, unbounded logits (measured range roughly -7.8 to +6.6 on this domain), not a [0,1] relevance score. The initial implementation compared these raw logits directly against `min_score=0.25` and silently returned empty results for *every* query, including genuinely relevant ones. Fixed by applying a sigmoid before thresholding — see `src/bayan/search/service.py`. Verified with a manual pair test: an identical query/text pair scores ~0.999 after sigmoid, a genuinely relevant pair ~0.15, and an unrelated pair ~0.0005 — real signal exists, it's just compressed into a much lower range than a typical similarity score, which is why `min_score` needed empirical tuning rather than a default guess.

## Lab 5 — Why labelled-query recall@10 is so low (data artifact, not a retrieval bug)
Initial full-corpus (20,000 cases) evaluation gave recall@10 ≈ 0.07 against the supplied `relevant_case_ids` — far below any reasonable target. Investigated by hand rather than assuming the retrieval code was broken:

1. Picked query Q-001 ("حفرة في الطريق أمام حي العليا منذ 16 أيام" — a pothole complaint) and looked up its 3 labelled `relevant_case_ids`. Their case texts: "الأسفلت متضرر قرب جدة..." (damaged asphalt near Jeddah), an English road-maintenance case near Al Yasmin, and a road case near An Narjis — **none textually or situationally close to the query**, beyond sharing the `roads` topic.
2. Checked the numeric pattern across the first 6 queries: relevant ids were `[1,9,17]`, `[10,18,26]`, `[19,27,35]`, `[28,36,44]`, `[37,45,53]`, `[46,54,62]` — **always the same topic, always spaced by exactly 8**. With 8 topics cycling round-robin through 20,000 rows, "every 8th same-topic row" is a mechanical artifact of how the case corpus was generated, not a semantic relevance judgement tied to the specific query text.
3. Added `topic_precision@10` (does each of the top-10 results share the query's topic?) as a fairer diagnostic: **1.0000** — every single top-10 result, for every one of the 130 answerable queries, was on-topic. The retrieval mechanism works; the label set just isn't a fair recall@10 yardstick for a content-based system, since it wasn't built from genuine top-relevance judgements.
4. Also added a "lenient" recall that credits a hit if the retrieved case shares its exact `case_text` with a labelled id (the corpus has heavy duplication: 20,000 rows, only 5,401 unique `case_text` values, some templates repeated 300+ times — average ~49 duplicates per labelled answer, up to 387). This raised recall slightly (0.069 → 0.077 without rerank) but not dramatically, because the labelled answers themselves usually aren't the closest semantic match to begin with (per point 1) — duplication wasn't the primary cause, the label-sampling method was.

Recorded honestly in `BENCHMARKS.md` rather than tuning the system to chase an unrepresentative metric. This is the same category of finding as the Lab 1 numbered-list gap, the Lab 3B incomplete QA smoke set, and the Lab 4 test-split language imbalance: a mismatch between what the course narrative assumes about a generated fixture and what the fixture actually contains.

## Lab 6 — The "parks→roads" investigation
Aggregate accuracy on `data/eval/validation_predictions.csv` was a suspiciously round 0.8750 (2,100/2,400) with a huge single-slice gap once sliced: `y_true=parks` scored **0.0000** (0/300) while every other topic scored a perfect 1.0000. Investigated rather than just reporting the number:

1. Pulled the actual feedback text (joined `feedback_id` back to `bayan_feedback.csv`) for a sample of the 300 errors. Most (e.g. "ألعاب الأطفال في حديقة حي العليا تحتاج صيانة" — playground equipment needs maintenance) share **no vocabulary with roads at all** — ruling out a simple keyword-overlap explanation as the primary cause. A minority (~15%, 45/300) do mention a street name like "طريق الملك فهد" as the location, which is a plausible partial contributor for those specific rows only.
2. Confirmed the failure is total and language-specific: 0/300 Arabic `parks` rows correct, and there are 0 English `parks` rows in this validation fixture at all (every other topic has both languages represented) — another instance of an uneven language/topic split in a generated fixture, similar to the Lab 4 test-split finding.
3. **Loaded the actual Lab 3A XLM-R classifier artefact and re-ran inference on the exact same 300 rows.** Result: **1.0000** — our real trained model gets every single one right. Ran `paired_bootstrap_diff` (our predictions vs the fixture's) over the full 2,400-row set: delta = **+0.1250**, 95% CI **[+0.1113, +0.1383]** — entirely positive, i.e. a real, statistically clear difference, not noise.
4. Conclusion: the `parks→roads` blind spot lives in whatever produced `validation_predictions.csv` (likely a deliberately planted teaching artifact for this exercise, or a stale/different model's output), **not** in the classifier this project actually trained and would ship. Documented as a new error-taxonomy category (#9 in `docs/ERROR_TAXONOMY.md`) since none of the 8 supplied categories describe a whole-slice, single-language, single-direction confusion like this, and reported honestly in `EVALUATION_REPORT.md` rather than either (a) silently trusting the fixture as if it were current production evidence, or (b) hiding the gap because "our" model doesn't have it.

## Lab 6 — Behavioural suite adaptation (no sentiment model exists)
`data/eval/behavioural_templates.csv` has 400 rows (200 `invariance` + 200 `directional`), built from only 2 unique sentence templates (one AR, one EN) with different incidental terms (city names, days, ...) substituted in. The `directional` rows' `expected_relation` column says "sentiment must not improve after negation" — but this project never built a dedicated sentiment model (Labs 1–5 only cover topic classification, NER, and QA), and the template itself is a fixed negative sentence regardless of which term is substituted, so there's no negation being added/removed to test directionally in the first place. `run_behavioural_suite()` therefore evaluates both test types with the same mechanism — majority-vote **topic**-consistency within each `(test_type, lang, template)` group — using the actual classifier this project has, and the "directional" rate is reported as a topic-robustness proxy, not a verified sentiment check. Measured (real classifier): invariance 60.0%, directional 80.0% — both below the course's reference targets (≥95% / ≥90%), a genuine, currently-unresolved robustness gap worth investigating further (unlike the parks/roads issue, this one **is** present in our actual model).

## Lab 7 — ONNX/INT8 export bugs and a real HTTP-load finding

**Bug found and fixed**: `ORTModelForSequenceClassification`/`ORTModelForTokenClassification` don't have a `.eval()` method (that's a `torch.nn.Module` thing; ONNX Runtime-wrapped models have no train/eval mode) — the first `export_onnx.py` draft called it unconditionally and crashed partway through, after the classifier ONNX/INT8 files had already been written successfully. Fixed by only calling `.eval()` on the plain torch path, and reused the already-exported classifier artefacts rather than wasting the earlier run.

**Noise caught in the single-request benchmark**: an initial n=200-sample run showed ONNX INT8 with the best p50 (7.72ms) but a *worse* p99 (46.37ms) than ONNX fp32 (15.39ms) — surprising, since quantised models are usually at least as fast. Rather than accept a counterintuitive result from a small sample, re-ran at n=500 with more warm-up: INT8 came back with p50=7.83ms **and** p99=12.51ms, better than every other rung. With only ~200 samples, p99 is essentially "the 2nd-highest value in the batch," extremely sensitive to a single OS-scheduling hiccup; n=500 is more stable. Verified the INT8 export was genuinely quantised (not silently loading the fp32 copy) by checking file sizes directly: `topic_classifier_onnx/model.onnx` = 1.06 GB vs `topic_classifier_int8/model_quantized.onnx` = 266 MB (~4x smaller, consistent with real INT8 weights).

**Real, unresolved finding — HTTP load test never meets the 40ms p99 target**: the single-request benchmark (INT8, p99=12.51ms) is well under the bare-inference 25ms target, but the 16-concurrent HTTP load test never got below 63ms p99 across three different thread-configuration attempts:
- No thread pinning in `api.py` at all → p99 = 113.94ms (each of 16 concurrent requests could spin up ONNX Runtime threads across all cores → severe oversubscription).
- `OMP_NUM_THREADS=4` + ONNX `intra_op_num_threads=4` → p99 = 63.41ms (best result; roughly halved).
- `intra_op_num_threads=1` (hypothesis: many single-threaded requests scheduled by the OS beats a few multi-threaded ones contending for cores) → p99 = 132.26ms, *worse* than no pinning at all. Reverted.

Concluded this is a genuine single-machine CPU capacity limit at 16-way concurrency for a 270M-parameter transformer — not something more thread-count tweaking was going to fix (already tried the two obvious directions). A real production fix would be horizontal scaling (multiple worker processes/replicas) or a smaller/distilled model, both out of scope here. Reported the actual measured numbers (63.41ms, missing the 40ms target) rather than quietly adjusting the target or only reporting the single-request number that does meet it.

**Tooling note**: `hey` (and the Go toolchain to build it) aren't available in this environment. Wrote `scripts/load_test.py`, a `ThreadPoolExecutor`-based equivalent that reproduces the same fixed-concurrency/fixed-duration load pattern and reports the same p50/p99/error summary, rather than downloading an untrusted prebuilt binary from the internet into an autonomous session.
