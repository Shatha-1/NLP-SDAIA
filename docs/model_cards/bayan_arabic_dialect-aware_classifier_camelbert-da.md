# Model Card — Bayan Arabic Dialect-Aware Classifier (CAMeLBERT-DA)

## Intended use
Arabic-only topic classification, chosen for Bayan's Gulf-majority Arabic traffic (Lab 4 dialect audit).

## Artefact / data versions
- Model/checkpoint: CAMeL-Lab/bert-base-arabic-camelbert-da, fine-tuned on Arabic-only data (Lab 4)
- Preprocessing version: 1.2.0
- Data version/snapshot: bayan_feedback.csv, Arabic subset with a citizen-grouped Gulf holdout (Lab 4)

## Metrics
- all/Gulf/MSA macro-F1: 1.0000 / 1.0000 / 1.0000 (tied with XLM-R and CAMeLBERT-mix; see BENCHMARKS.md Lab 4 for why accuracy alone doesn't break the tie)

## Slice metrics
See BENCHMARKS.md Lab 4 bake-off table (Gulf held-out slice carved from train; validation has 0 Gulf rows).

## Behavioural tests
Not evaluated in this report (behavioural suite above targets the XLM-R topic classifier only).

## Known limitations
- The tied 1.0000 macro-F1 across all three bake-off checkpoints means this synthetic benchmark cannot actually demonstrate a dialect-handling advantage numerically; the choice of CAMeLBERT-DA over CAMeLBERT-mix rests on tokenizer efficiency and Bayan's known traffic mix (Lab 4 dialect audit), not a measured accuracy win. On real, messier Gulf-dialect text the gap this model is meant to close may or may not materialise the same way.
- No artefact from the Colab bake-off run was persisted to this repository (`artifacts/` is gitignored and the run was ephemeral) — this card documents an evidence-backed decision, not a deployable checkpoint. Retraining would be needed before serving.
- Not evaluated against the behavioural suite or the same paired-comparison methodology used for the XLM-R card, so its practical robustness relative to XLM-R is unverified.

## Contact / owner
Shatha Hamad Bin Mana — SDA-AIE-211 course project (Bayan)