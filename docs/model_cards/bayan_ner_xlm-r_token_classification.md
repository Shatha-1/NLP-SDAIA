# Model Card — Bayan NER (XLM-R token classification)

## Intended use
Extract DATE/LOCATION/REFERENCE/SERVICE entities from citizen feedback.

## Artefact / data versions
- Model/checkpoint: xlm-roberta-base, fine-tuned (Lab 3B)
- Preprocessing version: 1.2.0
- Data version/snapshot: bayan_ner.conll (4,000 sentences)

## Metrics
- entity-F1 (frozen test): 1.0000 (see BENCHMARKS.md Lab 3 -- ceiling effect on templated synthetic data, documented there)

## Slice metrics
Not sliced in this report -- see BENCHMARKS.md Lab 4 for the LOCATION-recall segmentation comparison.

## Behavioural tests
Not applicable to NER in this project; behavioural suite above targets the topic classifier.

## Known limitations
- A perfect 1.0000 entity-F1 is a strong signal of an easy, templated synthetic corpus, not a claim of production robustness — every entity in `bayan_ner.conll` sits inside one of a small number of fixed sentence templates (see NOTES.md), so span boundaries are far more regular than free-form real citizen text would be.
- `ORGANISATION` is listed in the course's NER schema but never actually appears in the training/test data; this model has zero validated coverage for that entity type and should not be trusted on it without new labelled examples.
- No adversarial or out-of-template text was evaluated; performance on genuinely novel phrasing is unknown.

## Contact / owner
Shatha Hamad Bin Mana — SDA-AIE-211 course project (Bayan)