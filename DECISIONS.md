# Decision Records

## tokenizer
- Chosen checkpoint(s): `xlm-roberta-base` (XLM-R)
- Arabic fertility evidence: 1.672 subwords/word on 7,200 AR rows — second-lowest of the four candidates (only CAMeLBERT is lower, at 1.405) and with a 0.00% AR `<unk>` rate.
- English fertility evidence: 1.434 subwords/word on 4,800 EN rows — second-lowest of the four candidates (only DistilBERT is lower, at 1.298).
- p95 length evidence: 21 tokens (AR) / 23 tokens (EN) — the lowest combined p95 of all four candidates, meaning shorter, cheaper sequences for the same feedback text.
- Operational trade-off / rationale: Bayan is genuinely bilingual with code-switching in the same message (Latin case-reference codes inside Arabic sentences, mixed AR/EN feedback), so a single shared checkpoint must serve both languages well rather than being tuned to one. CAMeLBERT wins narrowly on pure Arabic fertility but is nearly 2x worse than XLM-R on English (2.705 vs 1.434), which would hurt the ~40% of feedback that is English or mixed. DistilBERT is English-only and unusable for Arabic (4.527 fertility, and it lowercases/strips diacritics, which is lossy). XLM-R is the best balanced choice — close-to-best on both languages individually, the lowest combined sequence length (cheaper training/serving), and zero UNK on Arabic thanks to its byte-level BPE vocabulary, so no signal is silently dropped for rare Arabic tokens.

## arabic-model
- Incumbent: `xlm-roberta-base` (Lab 3A multilingual classifier). All/Gulf/MSA macro-F1 = 1.0000 / 1.0000* / 1.0000 (*in-sample — already trained on all Gulf rows in Lab 3A, not a fair held-out number for that cell). AR fertility 1.589.
- Candidates: `CAMeL-Lab/bert-base-arabic-camelbert-mix` and `CAMeL-Lab/bert-base-arabic-camelbert-da`, each fine-tuned fresh on an Arabic-only training set with a genuine held-out Gulf slice (961 rows, citizen-grouped 20%, carved out of `train` because `validation` has zero Gulf rows) plus the MSA validation slice. Both scored all/Gulf/MSA = 1.0000/1.0000/1.0000, AR fertility 1.305 (identical to each other), trained in ~50s each on a Colab T4 GPU.
- All/Gulf/MSA evidence: all three checkpoints tie at a perfect 1.0000 macro-F1 on every slice — the measured Gulf-slice delta vs the incumbent is +0.00 for both candidates, not the course's +4 target. This is a genuine ceiling effect (same root cause as the Lab 3A/3B ceilings): the synthetic topic vocabulary is separable enough that dialect and pretraining corpus stop mattering for this task. Accuracy alone cannot break the tie here.
- CI-backed verdict: with accuracy tied, the decision falls back to two secondary, still-measured signals: (1) AR fertility — both CAMeLBERT variants tokenize Arabic 18% more efficiently than XLM-R (1.305 vs 1.589 subwords/word), meaning cheaper training/serving for the same Arabic text; (2) the Lab 4 dialect audit found Bayan's real Arabic traffic is 66.7% Gulf / 33.3% MSA, not MSA-dominant. Between the two CAMeLBERT variants (identical measured fertility and accuracy here), **CAMeLBERT-DA is the chosen candidate** — it's pretrained specifically on Dialectal Arabic, matching the majority of Bayan's actual traffic, whereas CAMeLBERT-mix's MSA-heavy pretraining mix is a worse match for that distribution even though this particular synthetic benchmark is too easy to show the gap numerically.
- Segmentation contract: clitic segmentation (`segment()`, CAMeL Tools D3 scheme) is evaluated independently in the NER path (see Lab 4 LOCATION recall note in `BENCHMARKS.md`) — it is not wired into this classification bake-off, since none of the topic-classification training/eval text here goes through `segment()`.

## search-min-score
- Threshold: **0.10**, applied to the cross-encoder score *after* a sigmoid transform (the raw `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` output is an unbounded logit, not a [0,1] similarity — measured range on this domain: roughly -7.8 to +6.6; even genuinely relevant short-text pairs often only reach ~0.10-0.20 after sigmoid, so this is not a conventional "0.5-ish similarity" cutoff).
- No-answer evidence: swept `min_score` over {0.02, 0.05, 0.10, 0.15, 0.20, 0.25} against all 150 labelled queries (20 of them genuinely unanswerable). All 20 no-answer queries were correctly rejected (empty result) at every threshold in that range, and answerable-query recall was identical at every threshold too — i.e. there's a wide, clean score gap between "on-topic case found" and "nothing relevant," so 0.10 is a robust middle-of-the-range choice, not a fragile knife-edge pick.
- False-positive / false-negative trade-off: because the sweep showed no trade-off inside the tested range (no threshold there caused an on-topic answer to be wrongly suppressed, nor a no-answer query to be wrongly served), the practical choice was to pick a value with headroom on both sides (0.10, roughly midway between the observed no-answer-query scores and the observed answerable-query scores) rather than sit right at either boundary of the tested range.

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
