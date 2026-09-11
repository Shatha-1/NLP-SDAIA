# Error Taxonomy

1. Label ambiguity
2. Arabic orthographic variation
3. Dialect or code-switching
4. Entity boundary or clitic alignment
5. Long-context truncation
6. Retrieval relevance mismatch
7. Preprocessing or serving skew
8. Annotation defect
9. **Arabic parks<->roads lexical confound** (added in Lab 6): a systematic, 100%-reproducible confusion where every Arabic `parks`-topic validation row in `data/eval/validation_predictions.csv` is predicted as `roads`, while 0 English `parks` rows show the same error. Read a sample of the underlying feedback text: most errors have no shared vocabulary with roads at all (e.g. "ألعاب الأطفال في حديقة حي العليا تحتاج صيانة" — playground equipment needs maintenance), so this looks like a language-specific blind spot baked into the reference/fixture prediction set itself, not a token- or entity-level confusion an existing category would capture. Confirmed distinct from our own trained classifier: re-running the same 300 rows through the actual Lab 3A XLM-R model scores 1.0000 on this exact slice (paired bootstrap delta vs the fixture: +0.125, 95% CI entirely positive) — see `NOTES.md` and `EVALUATION_REPORT.md`.
