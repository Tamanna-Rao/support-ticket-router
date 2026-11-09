"""
baseline.py
TF-IDF + Logistic Regression baseline on Banking77 (Syllabus: Unit I).

Run from the project root:
    python src/baseline.py
"""
import json
import os
import sys

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(__file__))
from preprocess import preprocess  # noqa: E402

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def load_banking77():
    """Load Banking77. Falls back to a mirror if the first source fails."""
    try:
        ds = load_dataset("PolyAI/banking77")
        label_names = ds["train"].features["label"].names
    except Exception as e:
        print("PolyAI/banking77 failed, trying mteb/banking77. Reason:", e)
        ds = load_dataset("mteb/banking77")
        pairs = dict(zip(ds["train"]["label"], ds["train"]["label_text"]))
        label_names = [pairs[i] for i in range(len(pairs))]
    return ds["train"], ds["test"], label_names


def run_experiment(name, X_train, y_train, X_test, y_test):
    """Train TF-IDF + LogReg and return the model, predictions, and metrics."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=10)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"[{name}] accuracy = {acc:.4f} | macro F1 = {macro_f1:.4f}")
    return pipe, preds, acc, macro_f1


def main():
    train, test, label_names = load_banking77()
    print(f"Train size: {len(train)} | Test size: {len(test)} | Classes: {len(label_names)}")

    counts = np.bincount(train["label"])
    print(f"Samples per class -> min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")

    y_train, y_test = train["label"], test["label"]

    # Experiment A: raw text (no preprocessing)
    _, _, acc_raw, f1_raw = run_experiment(
        "raw text", train["text"], y_train, test["text"], y_test
    )

    # Experiment B: with our preprocessing pipeline
    X_train_clean = [preprocess(t) for t in train["text"]]
    X_test_clean = [preprocess(t) for t in test["text"]]
    pipe, preds, acc_clean, f1_clean = run_experiment(
        "preprocessed", X_train_clean, y_train, X_test_clean, y_test
    )

    # Per-class report for the preprocessed model
    report = classification_report(
        y_test, preds, target_names=label_names, output_dict=True, zero_division=0
    )
    with open("results/baseline_report.json", "w") as f:
        json.dump(report, f, indent=2)

    summary = {
        "tfidf_logreg_raw": {"accuracy": acc_raw, "macro_f1": f1_raw},
        "tfidf_logreg_preprocessed": {"accuracy": acc_clean, "macro_f1": f1_clean},
    }
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    joblib.dump({"pipeline": pipe, "label_names": label_names}, "models/tfidf_logreg.joblib")

    worst = sorted(label_names, key=lambda n: report[n]["f1-score"])[:5]
    print("\nWeakest 5 intents (lowest F1):")
    for n in worst:
        print(f"  {n:45s} F1 = {report[n]['f1-score']:.3f}")

    print("\nSaved: models/tfidf_logreg.joblib, results/baseline_metrics.json, results/baseline_report.json")


if __name__ == "__main__":
    main()