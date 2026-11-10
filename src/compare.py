"""
compare.py
Builds the model comparison table, a bar chart, and an error analysis for DistilBERT.

Run from the project root (after all result files are in results/):
    python src/compare.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")  # save images instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score

RESULTS = "results"

MODEL_NAMES = {
    "tfidf_logreg_preprocessed": "TF-IDF + LogReg (preprocessed)",
    "tfidf_logreg_raw": "TF-IDF + LogReg (raw text)",
    "bilstm": "BiLSTM (random embeddings)",
    "bilstm_glove": "BiLSTM + GloVe",
    "distilbert": "DistilBERT (fine-tuned)",
}


def load_json(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def get_label_names():
    try:
        ds = load_dataset("PolyAI/banking77")
        return ds["train"].features["label"].names
    except Exception as e:
        print("PolyAI/banking77 failed, trying mteb/banking77. Reason:", e)
        ds = load_dataset("mteb/banking77")
        pairs = dict(zip(ds["train"]["label"], ds["train"]["label_text"]))
        return [pairs[i] for i in range(len(pairs))]


def build_table():
    metrics = {}
    for fname in ["baseline_metrics.json", "bilstm_metrics.json",
                  "bilstm_glove_metrics.json", "distilbert_metrics.json"]:
        metrics.update(load_json(fname))

    rows = [(MODEL_NAMES[k], metrics[k]) for k in MODEL_NAMES if k in metrics]
    lines = ["| Model | Accuracy | Macro F1 |", "|---|---|---|"]
    for name, m in rows:
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")
    table = "\n".join(lines)

    with open(os.path.join(RESULTS, "comparison_table.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table + "\n")

    # bar chart of macro F1
    names = [n for n, _ in rows][::-1]
    f1s = [m["macro_f1"] for _, m in rows][::-1]
    plt.figure(figsize=(9, 4))
    bars = plt.barh(names, f1s, color="#4C72B0")
    for bar, v in zip(bars, f1s):
        plt.text(v + 0.002, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center")
    plt.xlim(0.80, 0.96)
    plt.xlabel("Macro F1 on Banking77 test set (axis starts at 0.80)")
    plt.title("Model comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "model_comparison.png"), dpi=150)
    plt.close()
    print("Saved: results/comparison_table.md, results/model_comparison.png")


def error_analysis():
    preds_file = load_json("distilbert_test_preds.json")
    if not preds_file:
        print("Skipping error analysis (distilbert_test_preds.json not found).")
        return
    preds, gold = np.array(preds_file["preds"]), np.array(preds_file["gold"])
    label_names = get_label_names()
    n = len(label_names)

    cm = confusion_matrix(gold, preds, labels=list(range(n)))
    errors = cm.copy()
    np.fill_diagonal(errors, 0)

    pairs = [(errors[i, j], i, j) for i in range(n) for j in range(n) if errors[i, j] > 0]
    pairs.sort(reverse=True)

    per_class = f1_score(gold, preds, labels=list(range(n)), average=None, zero_division=0)
    worst = np.argsort(per_class)[:5]

    lines = ["# DistilBERT error analysis", "", "## Most confused intent pairs (true -> predicted)", ""]
    for count, i, j in pairs[:10]:
        lines.append(f"- {label_names[i]} -> {label_names[j]}: {count} times")
    lines += ["", "## Weakest 5 intents (lowest F1)", ""]
    for i in worst:
        lines.append(f"- {label_names[i]}: F1 = {per_class[i]:.3f}")
    text = "\n".join(lines)

    with open(os.path.join(RESULTS, "error_analysis.md"), "w") as f:
        f.write(text + "\n")
    print("\n" + text + "\n")

    # confusion matrix picture (colour scale capped so the errors are visible)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues", vmax=5)
    plt.title("DistilBERT confusion matrix (77 intents)")
    plt.xlabel("Predicted intent")
    plt.ylabel("True intent")
    plt.colorbar(label="Messages (capped at 5)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "distilbert_confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: results/error_analysis.md, results/distilbert_confusion_matrix.png")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    build_table()
    error_analysis()


if __name__ == "__main__":
    main()