"""Lab 6 starter: behavioural test generators/runners."""
from collections import Counter

import pandas as pd


def run_behavioural_suite(templates_df: pd.DataFrame, predict_fn) -> dict:
    """Run invariance/directional behavioural checks by filling {term} into each
    template and calling predict_fn(text) -> predicted label.

    Each (test_type, lang, template) group shares one underlying sentence with
    different incidental terms (city names, days, ...) substituted in. The
    majority-vote prediction across the group is treated as the "should stay
    unchanged" reference; each row's individual pass/fail is whether its own
    prediction matches that majority.

    Note: this project only trained a topic classifier (Labs 3-4), not a
    dedicated sentiment model, so "directional" rows (whose expected_relation
    in the source data is about sentiment, e.g. "sentiment must not improve
    after negation") are evaluated with the same topic-consistency mechanism
    as "invariance" rows, not a real sentiment check -- see NOTES.md.
    """
    records = []
    for (test_type, lang, template), group in templates_df.groupby(["test_type", "lang", "template"]):
        texts = [template.replace("{term}", str(term)) for term in group["term"]]
        preds = [predict_fn(text) for text in texts]
        majority_pred = Counter(preds).most_common(1)[0][0]

        for test_id, term, pred in zip(group["test_id"], group["term"], preds):
            records.append(
                {
                    "test_id": test_id,
                    "test_type": test_type,
                    "lang": lang,
                    "template": template,
                    "term": term,
                    "prediction": pred,
                    "majority_prediction": majority_pred,
                    "passed": pred == majority_pred,
                }
            )

    details = pd.DataFrame(records)
    rates = details.groupby("test_type")["passed"].mean().to_dict()
    return {"rates": rates, "details": details}
