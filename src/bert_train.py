"""
bert_train.py
Fine-tune DistilBERT on Banking77 (Syllabus: Unit V, HuggingFace Transformers).

Run on a GPU (Google Colab) from the project root:
    python src/bert_train.py
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


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


def make_items(tokenizer, texts, labels, max_len):
    enc = tokenizer(texts, truncation=True, max_length=max_len)
    return [
        {"input_ids": enc["input_ids"][i],
         "attention_mask": enc["attention_mask"][i],
         "labels": int(labels[i])}
        for i in range(len(texts))
    ]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds.extend(logits.argmax(1).cpu().tolist())
        golds.extend(labels.tolist())
    return np.array(preds), np.array(golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cpu":
        print("WARNING: no GPU found. Training on CPU will be very slow.")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train, test, label_names = load_banking77()
    texts, labels = list(train["text"]), list(train["label"])
    tr_x, va_x, tr_y, va_y = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = DataCollatorWithPadding(tokenizer)
    mk = lambda xs, ys, shuffle: DataLoader(
        make_items(tokenizer, xs, ys, args.max_len),
        batch_size=args.batch, shuffle=shuffle, collate_fn=collator,
    )
    train_dl, val_dl = mk(tr_x, tr_y, True), mk(va_x, va_y, False)
    test_dl = mk(list(test["text"]), list(test["label"]), False)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={n: i for i, n in enumerate(label_names)},
    ).to(device)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    use_amp = device.type == "cuda"  # mixed precision = faster on GPU
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_state = -1.0, None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(batch["labels"])
        v_pred, v_gold = predict(model, val_dl, device)
        v_f1 = f1_score(v_gold, v_pred, average="macro")
        print(f"Epoch {epoch:2d} | train loss {total / len(tr_x):.4f} | "
              f"val acc {accuracy_score(v_gold, v_pred):.4f} | val macro F1 {v_f1:.4f}")
        if v_f1 > best_f1:
            best_f1 = v_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    t_pred, t_gold = predict(model, test_dl, device)
    acc = accuracy_score(t_gold, t_pred)
    macro_f1 = f1_score(t_gold, t_pred, average="macro")
    print(f"\n[distilbert] TEST accuracy = {acc:.4f} | macro F1 = {macro_f1:.4f} "
          f"| train time {time.time() - start:.0f}s")

    with open("results/distilbert_metrics.json", "w") as f:
        json.dump({"distilbert": {"accuracy": acc, "macro_f1": macro_f1}}, f, indent=2)
    with open("results/distilbert_test_preds.json", "w") as f:
        json.dump({"preds": t_pred.tolist(), "gold": t_gold.tolist()}, f)

    model.save_pretrained("models/distilbert")
    tokenizer.save_pretrained("models/distilbert")
    print("Saved: results/distilbert_metrics.json, results/distilbert_test_preds.json, models/distilbert/")


if __name__ == "__main__":
    main()