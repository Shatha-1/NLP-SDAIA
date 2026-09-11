"""Lab 6 starter: generate EVALUATION_REPORT.md + model-card evidence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import torch
from jinja2 import Template
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bayan.evaluation.behavioural import run_behavioural_suite
from bayan.evaluation.bootstrap import bootstrap_ci, paired_bootstrap_diff
from bayan.evaluation.slices import sliced_report
from bayan.preprocessing.core import PREPROC_VERSION, preprocess

PREDICTIONS_PATH = Path("data/eval/validation_predictions.csv")
FEEDBACK_PATH = Path("data/raw/bayan_feedback.csv")
BEHAVIOURAL_PATH = Path("data/eval/behavioural_templates.csv")
CLASSIFIER_DIR = "artifacts/topic_classifier"
MODEL_CARD_TEMPLATE = Path("templates/model_card.md.j2")
MODEL_CARDS_DIR = Path("docs/model_cards")


def _load_classifier_predict_fn():
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_DIR)
    model.eval()
    id2label = model.config.id2label

    def predict(text: str) -> str:
        inputs = tokenizer(preprocess(text), return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = model(**inputs).logits
        return id2label[int(logits.argmax(dim=-1))]

    return predict


def _categorise_error(row) -> str:
    if row["y_true"] == "parks" and row["y_pred"] == "roads" and row["lang"] == "ar":
        return "Arabic parks<->roads lexical confound (new category, see below)"
    return "Uncategorised"


def _df_to_markdown(df: pd.DataFrame, float_cols=()) -> str:
    df = df.copy()
    for col in float_cols:
        df[col] = df[col].map(lambda x: f"{x:.4f}")
    return df.to_markdown(index=False)


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    correct = (df["y_true"] == df["y_pred"]).astype(int).tolist()
    agg_point, agg_lo, agg_hi = bootstrap_ci(correct)
    print(f"Aggregate accuracy: {agg_point:.4f} [{agg_lo:.4f}, {agg_hi:.4f}]")

    slices = sliced_report(df, ["lang", "dialect_region", "y_true", "length_bucket"])
    print("\nSliced report (flagging n < 30):")
    print(slices.to_string(index=False))
    small = slices[slices["small_slice"]]
    print(f"\n{len(small)} slice(s) flagged too small to trust (n < 30).")

    feedback = pd.read_csv(FEEDBACK_PATH)[["feedback_id", "text"]]
    joined = df.merge(feedback, on="feedback_id", how="left")
    predict = _load_classifier_predict_fn()
    joined["y_pred_ours"] = joined["text"].map(predict)

    fixture_correct = (joined["y_true"] == joined["y_pred"]).astype(int).tolist()
    ours_correct = (joined["y_true"] == joined["y_pred_ours"]).astype(int).tolist()
    delta, d_lo, d_hi = paired_bootstrap_diff(ours_correct, fixture_correct)
    print(
        f"\nPaired comparison (our XLM-R classifier vs fixture predictions): "
        f"delta={delta:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}]"
    )

    parks_ar = joined[(joined["y_true"] == "parks") & (joined["lang"] == "ar")]
    ours_parks_acc = (parks_ar["y_pred_ours"] == "parks").mean()
    print(f"Our classifier's accuracy on the Arabic 'parks' blind-spot slice: {ours_parks_acc:.4f}")

    errors = joined[joined["y_true"] != joined["y_pred"]].copy()
    errors["category"] = errors.apply(_categorise_error, axis=1)
    taxonomy_counts = errors["category"].value_counts().to_dict()
    print(f"\nError taxonomy (n={len(errors)}):")
    for cat, n in taxonomy_counts.items():
        print(f"  {cat}: {n} ({n / len(errors):.1%})")

    behavioural_df = pd.read_csv(BEHAVIOURAL_PATH)
    behaviour = run_behavioural_suite(behavioural_df, predict)
    print("\nBehavioural rates:")
    for test_type, rate in behaviour["rates"].items():
        print(f"  {test_type}: {rate:.1%}")

    # --- Model cards ---
    MODEL_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    template = Template(MODEL_CARD_TEMPLATE.read_text(encoding="utf-8"))

    slices_md = _df_to_markdown(slices, float_cols=["accuracy", "ci_lo", "ci_hi"])
    behavioural_md = "\n".join(f"- {t}: {r:.1%}" for t, r in behaviour["rates"].items())

    cards = [
        {
            "model_name": "Bayan Topic Classifier (XLM-R)",
            "intended_use": "Route bilingual (AR/EN) citizen feedback to one of 8 service topics.",
            "checkpoint": "xlm-roberta-base, fine-tuned (Lab 3A)",
            "preproc_version": PREPROC_VERSION,
            "data_version": "bayan_feedback.csv (12,000 rows)",
            "metrics_table": (
                f"- Aggregate accuracy (validation_predictions.csv): {agg_point:.4f} "
                f"[{agg_lo:.4f}, {agg_hi:.4f}]\n"
                f"- Paired delta vs this fixture's reference predictions: {delta:+.4f} "
                f"[{d_lo:+.4f}, {d_hi:+.4f}]"
            ),
            "slices_table": slices_md,
            "behavioural_table": behavioural_md,
        },
        {
            "model_name": "Bayan NER (XLM-R token classification)",
            "intended_use": "Extract DATE/LOCATION/REFERENCE/SERVICE entities from citizen feedback.",
            "checkpoint": "xlm-roberta-base, fine-tuned (Lab 3B)",
            "preproc_version": PREPROC_VERSION,
            "data_version": "bayan_ner.conll (4,000 sentences)",
            "metrics_table": "- entity-F1 (frozen test): 1.0000 (see BENCHMARKS.md Lab 3 -- ceiling effect on templated synthetic data, documented there)",
            "slices_table": "Not sliced in this report -- see BENCHMARKS.md Lab 4 for the LOCATION-recall segmentation comparison.",
            "behavioural_table": "Not applicable to NER in this project; behavioural suite above targets the topic classifier.",
        },
        {
            "model_name": "Bayan Arabic Dialect-Aware Classifier (CAMeLBERT-DA)",
            "intended_use": "Arabic-only topic classification, chosen for Bayan's Gulf-majority Arabic traffic (Lab 4 dialect audit).",
            "checkpoint": "CAMeL-Lab/bert-base-arabic-camelbert-da, fine-tuned on Arabic-only data (Lab 4)",
            "preproc_version": PREPROC_VERSION,
            "data_version": "bayan_feedback.csv, Arabic subset with a citizen-grouped Gulf holdout (Lab 4)",
            "metrics_table": "- all/Gulf/MSA macro-F1: 1.0000 / 1.0000 / 1.0000 (tied with XLM-R and CAMeLBERT-mix; see BENCHMARKS.md Lab 4 for why accuracy alone doesn't break the tie)",
            "slices_table": "See BENCHMARKS.md Lab 4 bake-off table (Gulf held-out slice carved from train; validation has 0 Gulf rows).",
            "behavioural_table": "Not evaluated in this report (behavioural suite above targets the XLM-R topic classifier only).",
        },
    ]

    for card in cards:
        rendered = template.render(**card)
        out_path = MODEL_CARDS_DIR / (card["model_name"].lower().replace(" ", "_").replace("(", "").replace(")", "") + ".md")
        out_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote model card: {out_path}")

    print("\nUpdate EVALUATION_REPORT.md by hand using the numbers above (known limitations must be written manually).")


if __name__ == "__main__":
    main()
