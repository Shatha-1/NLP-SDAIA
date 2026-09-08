"""Lab 3A starter: TF-IDF + LinearSVC baseline."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC

from bayan.models.data import build_topic_dataset
from bayan.preprocessing.core import preprocess


def main():
    ds = build_topic_dataset()

    train_texts = ds["train"]["text"].map(preprocess)
    val_texts = ds["validation"]["text"].map(preprocess)
    test_texts = ds["test"]["text"].map(preprocess)

    train_labels = ds["train"]["topic"]
    val_labels = ds["validation"]["topic"]
    test_labels = ds["test"]["topic"]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LinearSVC(random_state=42)
    clf.fit(X_train, train_labels)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_f1 = f1_score(val_labels, val_pred, average="macro")
    test_f1 = f1_score(test_labels, test_pred, average="macro")

    print("TF-IDF + LinearSVC baseline")
    print(f"  train={len(train_labels)}  validation={len(val_labels)}  test={len(test_labels)}")
    print(f"  Validation macro-F1:  {val_f1:.4f}")
    print(f"  Frozen test macro-F1: {test_f1:.4f}")
    print()
    print("Validation classification report:")
    print(classification_report(val_labels, val_pred, zero_division=0))
    print("Record these numbers in BENCHMARKS.md under 'Lab 3 - Models'.")

    return {"validation_macro_f1": val_f1, "test_macro_f1": test_f1}


if __name__ == "__main__":
    main()
