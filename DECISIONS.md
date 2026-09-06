# Decision Records

## tokenizer
- Chosen checkpoint(s): `xlm-roberta-base` (XLM-R)
- Arabic fertility evidence: 1.672 subwords/word on 7,200 AR rows — second-lowest of the four candidates (only CAMeLBERT is lower, at 1.405) and with a 0.00% AR `<unk>` rate.
- English fertility evidence: 1.434 subwords/word on 4,800 EN rows — second-lowest of the four candidates (only DistilBERT is lower, at 1.298).
- p95 length evidence: 21 tokens (AR) / 23 tokens (EN) — the lowest combined p95 of all four candidates, meaning shorter, cheaper sequences for the same feedback text.
- Operational trade-off / rationale: Bayan is genuinely bilingual with code-switching in the same message (Latin case-reference codes inside Arabic sentences, mixed AR/EN feedback), so a single shared checkpoint must serve both languages well rather than being tuned to one. CAMeLBERT wins narrowly on pure Arabic fertility but is nearly 2x worse than XLM-R on English (2.705 vs 1.434), which would hurt the ~40% of feedback that is English or mixed. DistilBERT is English-only and unusable for Arabic (4.527 fertility, and it lowercases/strips diacritics, which is lossy). XLM-R is the best balanced choice — close-to-best on both languages individually, the lowest combined sequence length (cheaper training/serving), and zero UNK on Arabic thanks to its byte-level BPE vocabulary, so no signal is silently dropped for rare Arabic tokens.

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
