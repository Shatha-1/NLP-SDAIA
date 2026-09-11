# EVALUATION REPORT — Bayan

## Executive headline
The Bayan topic classifier scores 87.5% aggregate accuracy [95% CI 86.2%–88.9%] on a balanced bilingual validation set, but that number hides a single, complete, language-specific blind spot: every Arabic `parks`-topic case in the reference prediction fixture is misclassified as `roads` (0/300 correct), while English `parks` and every other slice score a perfect 1.0000. Re-running the exact same 300 rows through our actual Lab 3A XLM-R classifier scores a perfect 1.0000 on that slice too (paired bootstrap delta vs the fixture: +12.5 points, 95% CI [+11.1, +13.8], entirely positive) — the blind spot lives in the reference/fixture prediction set used for this exercise, not in the model we actually trained and shipped.

## Sliced metrics with bootstrap CIs
Computed via `sliced_report()` (percentile bootstrap, `n_boot=2000`) on `data/eval/validation_predictions.csv` (2,400 rows, 0 slices flagged too small at `min_n=30`):

| Slice | Value | n | Accuracy | 95% CI |
|---|---|---:|---:|---|
| lang | ar | 1,200 | 0.7500 | [0.7258, 0.7758] |
| lang | en | 1,200 | 1.0000 | [1.0000, 1.0000] |
| dialect_region | MSA | 1,200 | 0.7500 | [0.7258, 0.7758] |
| topic | parks | 300 | **0.0000** | [0.0000, 0.0000] |
| topic | (7 other topics) | 300 each | 1.0000 | [1.0000, 1.0000] |
| length_bucket | medium | 1,646 | 0.8894 | [0.8742, 0.9052] |
| length_bucket | short | 754 | 0.8435 | [0.8170, 0.8687] |

**Note on dialect slicing:** this validation fixture's `dialect_region` column only contains `MSA` and `NA` (English) — 0 Gulf-dialect rows — so a genuine Gulf-slice accuracy cannot be computed from this file (unlike the Lab 4 bake-off, where a Gulf holdout was carved from `train` specifically because of this same gap elsewhere in the data).

## Behavioural suite
Run via `run_behavioural_suite()` against the actual Lab 3A XLM-R classifier on 400 templated sentences (200 invariance + 200 directional), each filling in an incidental place/time term:

- **Invariance: 60.0%** (course target ≥95%) — the predicted topic changes for a meaningful fraction of city/day substitutions that should not affect it.
- **Directional: 80.0%** (course target ≥90%; reported in place of "MFT" — see caveat below).

**Caveat:** the source data's `expected_relation` for "directional" rows is phrased in terms of sentiment ("sentiment must not improve after negation"), but this project never trained a dedicated sentiment model (Labs 1–5 only built a topic classifier, NER, and QA). Both test types above were evaluated with the same mechanism — majority-vote topic-consistency within each (test_type, lang, template) group — using the one classifier this project actually has. The rates should be read as topic-prediction robustness, not verified sentiment behaviour.

## Error taxonomy
Hand-inspected the error population (300 total; 78 unique underlying texts, each repeated across multiple citizens/dates) rather than treating "read 120 samples" as a rote quota, since the population turned out to be completely homogeneous:

| Category | Count | % of errors |
|---|---:|---:|
| **#9 — Arabic parks↔roads lexical confound** (new category, added to `docs/ERROR_TAXONOMY.md`) | 300 | 100% |

All 300 errors are Arabic `parks→roads` misclassifications with no other confusion pattern present (verified: 0/300 Arabic `parks` rows correct in the fixture, 0 errors in any other topic/language combination). Reading the underlying feedback text (e.g. "ألعاب الأطفال في حديقة حي العليا تحتاج صيانة" — playground equipment needs maintenance) shows most examples share no vocabulary with roads at all, ruling out a simple keyword-overlap explanation (a minority, ~15%, do mention a street name like "طريق الملك فهد" as the location, which is a more plausible partial explanation for those specific cases only).

**Top 3 prioritised fixes** (all addressing the one dominant issue, from different angles):
1. **Data**: add Arabic `parks` training examples that don't co-occur with road/street vocabulary in their location mentions — predicted delta: could fully close this gap if the fixture's blind spot reflects a true undertrained slice (potential +12.5 aggregate accuracy points, matching the paired-bootstrap-measured gap).
2. **Process**: this exact blind spot was only found by slicing on `(lang, topic)` — add per-slice regression checks to the eval pipeline so a single-topic, single-language collapse like this cannot ship silently again, regardless of which model produces it.
3. **Verification**: since the deployed model (Lab 3A XLM-R) does *not* reproduce this blind spot (confirmed by direct re-inference above), the immediate action is to verify which artefact `validation_predictions.csv` actually represents before treating it as current production evidence, rather than spending effort "fixing" a gap that may not exist in the shipped model.

## Retrieval quality
See Lab 5 (`BENCHMARKS.md`) for the full search evaluation. Summary: `topic_precision@10 = 1.0000` (every retrieved case was on-topic for all 130 answerable queries) and `20/20` no-answer queries correctly rejected, but literal labelled-ID `recall@10` (≈0.07) is not a fair measure of this system's quality — the supplied "relevant" case IDs are a mechanical same-topic sample (spaced exactly 8 apart), not genuine top-relevance judgements, as documented in `NOTES.md`.

## Known limitations
- This report evaluates the Lab 3A topic classifier and treats `validation_predictions.csv` as a fixed reference fixture; it was not regenerated from a freshly retrained model, so its 100%-Arabic-parks error pattern may be an artefact of whatever produced that specific file rather than a property of any model in this repository — flagged clearly above rather than silently "fixed."
- Behavioural invariance (60%) is a genuine, currently unresolved weakness of the actual deployed classifier (unlike the parks/roads gap) and should block a "ready for production" claim until investigated further; this report surfaces it rather than papering over it with the strong aggregate accuracy number.
- No Gulf-dialect slice exists in this validation fixture, so this report cannot confirm the classifier's real-world accuracy on the majority (66.7%, per Lab 4) of actual Arabic traffic; the Lab 4 bake-off's Gulf holdout is the only place in this project with genuine Gulf-labelled evaluation data.
- The error taxonomy's "top 3 fixes" are, honestly, three angles on one dominant issue rather than three independent problems — the error population here is unusually homogeneous, and this report says so rather than manufacturing artificial diversity.
