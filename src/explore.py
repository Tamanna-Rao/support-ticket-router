"""
explore.py
Quick exploratory data analysis (EDA) on Banking77.

Run from the project root:
    python src/explore.py
"""
import os

import matplotlib
matplotlib.use("Agg")  # save charts as images instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

os.makedirs("results", exist_ok=True)


def load_banking77():
    try:
        ds = load_dataset("PolyAI/banking77")
        label_names = ds["train"].features["label"].names
    except Exception as e:
        print("PolyAI/banking77 failed, trying mteb/banking77. Reason:", e)
        ds = load_dataset("mteb/banking77")
        pairs = dict(zip(ds["train"]["label"], ds["train"]["label_text"]))
        label_names = [pairs[i] for i in range(len(pairs))]
    return ds["train"], ds["test"], label_names


def main():
    train, test, label_names = load_banking77()
    texts = train["text"]
    labels = np.array(train["label"])

    # 1. Basic facts
    print(f"Training messages: {len(train)}")
    print(f"Test messages:     {len(test)}")
    print(f"Number of intents: {len(label_names)}")

    # 2. Message length (in words)
    lengths = np.array([len(t.split()) for t in texts])
    print(f"\nMessage length in words -> min: {lengths.min()}, "
          f"max: {lengths.max()}, mean: {lengths.mean():.1f}, "
          f"median: {np.median(lengths):.0f}")
    print(f"95% of messages are {int(np.percentile(lengths, 95))} words or shorter")

    # 3. Class balance
    counts = np.bincount(labels, minlength=len(label_names))
    order = np.argsort(counts)
    print("\n5 smallest classes:")
    for i in order[:5]:
        print(f"  {label_names[i]:45s} {counts[i]}")
    print("5 largest classes:")
    for i in order[-5:][::-1]:
        print(f"  {label_names[i]:45s} {counts[i]}")

    # 4. Sample messages
    print("\nSample messages:")
    rng = np.random.default_rng(42)
    for idx in rng.choice(len(train), size=8, replace=False):
        print(f"  [{label_names[labels[idx]]}] {texts[idx]}")

    # Chart 1: message length histogram
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=range(0, lengths.max() + 2), color="#4C72B0", edgecolor="white")
    plt.title("Message length distribution (words)")
    plt.xlabel("Words per message")
    plt.ylabel("Number of messages")
    plt.tight_layout()
    plt.savefig("results/length_distribution.png", dpi=150)
    plt.close()

    # Chart 2: samples per class (all 77, sorted)
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(counts)), counts[order], color="#55A868")
    plt.title("Training samples per intent (sorted)")
    plt.xlabel("Intent (77 classes)")
    plt.ylabel("Number of messages")
    plt.xticks([])
    plt.tight_layout()
    plt.savefig("results/class_distribution.png", dpi=150)
    plt.close()

    print("\nSaved charts: results/length_distribution.png, results/class_distribution.png")


if __name__ == "__main__":
    main()