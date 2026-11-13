"""
threshold_analysis.py
Picks the router's confidence thresholds from VALIDATION data, checks them on the TEST set,
measures team-level accuracy, and estimates the time/cost saved.

Run from the project root (CPU is fine, takes a few minutes):
    python src/threshold_analysis.py
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from router import MODEL_ID, team_for  # noqa: E402

RESULTS = "results"

# ---- choices and ASSUMPTIONS (edit these; they are not measured, they are examples) ----
TARGET_AUTO_ACCURACY = 0.97   # auto-routed tickets should be right at least this often (on validation)
REVIEW_THRESHOLD = 0.50       # below this, a human triages the ticket
TICKETS_PER_MONTH = 10_000
MANUAL_MIN = 2.0              # minutes for a human to triage one ticket by hand
REVIEW_MIN = 0.5              # minutes to quickly check a "needs review" ticket
REWORK_MIN = 10.0             # extra minutes to fix a ticket sent to the wrong team
AGENT_COST_PER_HOUR = 300     # INR per hour


# ------------------------------------------------------------------ pure functions
def sweep(conf, intent_ok, team_ok, thresholds):
    """For each threshold: (threshold, share auto-routed, intent accuracy, team accuracy)."""
    rows = []
    for t in thresholds:
        mask = conf >= t
        n = int(mask.sum())
        rows.append((float(t), n / len(conf),
                     float(intent_ok[mask].mean()) if n else float("nan"),
                     float(team_ok[mask].mean()) if n else float("nan")))
    return rows


def choose_auto_threshold(rows, target, min_share=0.05):
    """Smallest threshold whose auto-routed intent accuracy reaches the target."""
    for t, share, acc, _ in rows:
        if share >= min_share and acc >= target:
            return t, True
    best = max((r for r in rows if r[1] >= min_share), key=lambda r: r[2])
    return best[0], False


def band_stats(conf, intent_ok, team_ok, auto_t, review_t):
    bands = {
        "auto-route": conf >= auto_t,
        "route + needs review": (conf >= review_t) & (conf < auto_t),
        "human triage": conf < review_t,
    }
    out = {}
    for name, m in bands.items():
        n = int(m.sum())
        out[name] = {
            "share": n / len(conf),
            "intent_acc": float(intent_ok[m].mean()) if n else float("nan"),
            "team_acc": float(team_ok[m].mean()) if n else float("nan"),
        }
    return out


def estimate_savings(stats):
    n_auto = stats["auto-route"]["share"] * TICKETS_PER_MONTH
    n_review = stats["route + needs review"]["share"] * TICKETS_PER_MONTH
    wrong_team_auto = n_auto * (1 - stats["auto-route"]["team_acc"]) if n_auto else 0.0
    baseline_min = TICKETS_PER_MONTH * MANUAL_MIN
    saved_min = (n_auto * MANUAL_MIN + n_review * (MANUAL_MIN - REVIEW_MIN)
                 - wrong_team_auto * REWORK_MIN)
    return {
        "baseline_hours": baseline_min / 60,
        "saved_hours": saved_min / 60,
        "saved_pct": 100 * saved_min / baseline_min,
        "saved_inr": saved_min / 60 * AGENT_COST_PER_HOUR,
        "wrong_team_auto": wrong_team_auto,
    }


# ------------------------------------------------------------------ model + data
def load_data():
    from datasets import load_dataset

    try:
        ds = load_dataset("PolyAI/banking77")
        label_names = ds["train"].features["label"].names
    except Exception as e:
        print("PolyAI/banking77 failed, trying mteb/banking77. Reason:", e)
        ds = load_dataset("mteb/banking77")
        pairs = dict(zip(ds["train"]["label"], ds["train"]["label_text"]))
        label_names = [pairs[i] for i in range(len(pairs))]
    return ds["train"], ds["test"], label_names


def run_model(model, tokenizer, texts, batch=64):
    import torch

    confs, preds = [], []
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i:i + batch], truncation=True, max_length=64,
                        padding=True, return_tensors="pt")
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        c, p = probs.max(dim=-1)
        confs += c.tolist()
        preds += p.tolist()
        if (i // batch) % 10 == 0:
            print(f"  {min(i + batch, len(texts))}/{len(texts)}")
    return np.array(confs), np.array(preds)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.makedirs(RESULTS, exist_ok=True)
    train, test, label_names = load_data()
    team_of = np.array([team_for(n) for n in label_names])

    # same validation split as in bert_train.py (10%, stratified, seed 42)
    _, va_x, _, va_y = train_test_split(
        list(train["text"]), list(train["label"]),
        test_size=0.1, stratify=list(train["label"]), random_state=42)
    te_x, te_y = list(test["text"]), np.array(test["label"])
    va_y = np.array(va_y)

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

    print("Scoring validation set...")
    va_conf, va_pred = run_model(model, tokenizer, va_x)
    print("Scoring test set...")
    te_conf, te_pred = run_model(model, tokenizer, te_x)

    va_ok, te_ok = va_pred == va_y, te_pred == te_y
    va_team = team_of[va_pred] == team_of[va_y]
    te_team = team_of[te_pred] == team_of[te_y]

    # 1) choose the auto threshold on validation
    thresholds = np.round(np.arange(0.50, 1.0, 0.01), 2)
    va_rows = sweep(va_conf, va_ok, va_team, thresholds)
    auto_t, reached = choose_auto_threshold(va_rows, TARGET_AUTO_ACCURACY)

    # 2) report on the untouched test set
    stats = band_stats(te_conf, te_ok, te_team, auto_t, REVIEW_THRESHOLD)
    money = estimate_savings(stats)

    lines = [
        "# Routing analysis",
        "",
        f"Thresholds chosen on the validation set (target: {TARGET_AUTO_ACCURACY:.0%} intent accuracy "
        f"on auto-routed tickets{'' if reached else ' - NOT reached, using the best available'}):",
        "",
        f"- Auto-route at confidence >= **{auto_t:.2f}**",
        f"- Human triage below **{REVIEW_THRESHOLD:.2f}**",
        "",
        "## Test-set results with these thresholds",
        "",
        f"Overall intent accuracy: {te_ok.mean():.4f} | overall team accuracy: {te_team.mean():.4f} "
        f"| mean confidence: {te_conf.mean():.4f}",
        "",
        "| Decision | Share of tickets | Intent accuracy | Team accuracy |",
        "|---|---|---|---|",
    ]
    for name, s in stats.items():
        lines.append(f"| {name} | {s['share']:.1%} | {s['intent_acc']:.3f} | {s['team_acc']:.3f} |")
    lines += [
        "",
        "## Estimated monthly impact (ASSUMPTIONS, not measurements)",
        "",
        f"Assumptions: {TICKETS_PER_MONTH:,} tickets/month; {MANUAL_MIN} min manual triage; "
        f"{REVIEW_MIN} min to check a flagged ticket; {REWORK_MIN} min to fix a wrong-team ticket; "
        f"INR {AGENT_COST_PER_HOUR}/hour.",
        "",
        f"- Manual triage today: {money['baseline_hours']:.0f} hours/month",
        f"- Estimated net time saved: {money['saved_hours']:.0f} hours/month ({money['saved_pct']:.0f}%)",
        f"- Estimated net saving: INR {money['saved_inr']:,.0f}/month",
        f"- Auto-routed tickets sent to the wrong team: about {money['wrong_team_auto']:.0f}/month",
    ]
    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "routing_analysis.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(RESULTS, "thresholds.json"), "w") as f:
        json.dump({"auto": auto_t, "review": REVIEW_THRESHOLD}, f, indent=2)
    print("\n" + text + "\n")

    # 3) trade-off chart (validation curves, test result marked)
    shares = [r[1] for r in va_rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(shares, [r[2] for r in va_rows], label="Intent accuracy (validation)")
    plt.plot(shares, [r[3] for r in va_rows], label="Team accuracy (validation)")
    s = stats["auto-route"]
    plt.scatter([s["share"]], [s["intent_acc"]], color="red", zorder=5,
                label=f"Chosen threshold {auto_t:.2f} (test set)")
    plt.xlabel("Share of tickets auto-routed")
    plt.ylabel("Accuracy of auto-routed tickets")
    plt.title("Automation vs accuracy trade-off")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "threshold_tradeoff.png"), dpi=150)
    plt.close()
    print("Saved: results/routing_analysis.md, results/thresholds.json, results/threshold_tradeoff.png")


if __name__ == "__main__":
    main()