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
| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | | | |
| CAMeLBERT | | | |

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
