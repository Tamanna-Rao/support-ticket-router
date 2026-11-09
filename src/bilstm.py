"""
bilstm.py
BiLSTM intent classifier on Banking77 (Syllabus: Unit III, with GloVe from Unit II).

Best run on a GPU (Google Colab). From the project root:
    python src/bilstm.py            # random embeddings
    python src/bilstm.py --glove    # GloVe-initialised embeddings
"""
import argparse
import json
import os
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD, UNK = 0, 1


# ----------------------------------------------------------------- data
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


def tokenize(text):
    """Lowercase and split into word tokens (keeps apostrophes, e.g. "hasn't")."""
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab(texts):
    counter = Counter(tok for t in texts for tok in tokenize(t))
    itos = ["<pad>", "<unk>"] + sorted(counter)
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def encode(text, stoi):
    ids = [stoi.get(tok, UNK) for tok in tokenize(text)]  # unseen words -> <unk> (OOV handling)
    return ids if ids else [UNK]


class TextDataset(Dataset):
    def __init__(self, texts, labels, stoi):
        self.ids = [torch.tensor(encode(t, stoi), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD)
    return padded, lengths, torch.stack(labels)


# ----------------------------------------------------------------- model
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, n_classes, dropout=0.4, emb_matrix=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        if emb_matrix is not None:
            self.emb.weight.data.copy_(emb_matrix)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x, lengths):
        emb = self.drop(self.emb(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        feats = torch.cat([h[-2], h[-1]], dim=1)  # last forward + last backward state
        return self.fc(self.drop(feats))


def load_glove_matrix(itos, dim):
    import gensim.downloader as api

    print(f"Loading GloVe {dim}d (downloads ~130 MB the first time)...")
    glove = api.load(f"glove-wiki-gigaword-{dim}")
    matrix = np.random.normal(0, 0.1, (len(itos), dim)).astype("float32")
    matrix[PAD] = 0
    hits = 0
    for i, w in enumerate(itos):
        if w in glove:
            matrix[i] = glove[w]
            hits += 1
    print(f"GloVe covers {hits}/{len(itos)} vocabulary words ({100 * hits / len(itos):.1f}%)")
    return torch.tensor(matrix)


# ----------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, golds = [], []
    for x, lengths, y in loader:
        logits = model(x.to(device), lengths)
        preds.extend(logits.argmax(1).cpu().tolist())
        golds.extend(y.tolist())
    return np.array(preds), np.array(golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", action="store_true", help="initialise embeddings with GloVe")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train, test, label_names = load_banking77()
    texts, labels = list(train["text"]), list(train["label"])

    # hold out 10% of train as validation (used to pick the best epoch)
    tr_x, va_x, tr_y, va_y = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=args.seed
    )
    itos, stoi = build_vocab(tr_x)  # vocabulary from training data only
    print(f"Vocabulary size: {len(itos)}")

    mk = lambda xs, ys, shuffle: DataLoader(
        TextDataset(xs, ys, stoi), batch_size=args.batch, shuffle=shuffle, collate_fn=collate
    )
    train_dl, val_dl = mk(tr_x, tr_y, True), mk(va_x, va_y, False)
    test_dl = mk(list(test["text"]), list(test["label"]), False)

    emb_matrix = load_glove_matrix(itos, args.emb) if args.glove else None
    model = BiLSTMClassifier(
        len(itos), args.emb, args.hidden, len(label_names), emb_matrix=emb_matrix
    ).to(device)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_state, bad_epochs = -1.0, None, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, lengths, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(y)
        v_pred, v_gold = predict(model, val_dl, device)
        v_f1 = f1_score(v_gold, v_pred, average="macro")
        print(f"Epoch {epoch:2d} | train loss {total / len(tr_x):.4f} | "
              f"val acc {accuracy_score(v_gold, v_pred):.4f} | val macro F1 {v_f1:.4f}")
        if v_f1 > best_f1:
            best_f1, bad_epochs = v_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs)")
                break

    model.load_state_dict(best_state)
    t_pred, t_gold = predict(model, test_dl, device)
    acc = accuracy_score(t_gold, t_pred)
    macro_f1 = f1_score(t_gold, t_pred, average="macro")
    name = "bilstm_glove" if args.glove else "bilstm"
    print(f"\n[{name}] TEST accuracy = {acc:.4f} | macro F1 = {macro_f1:.4f} "
          f"| train time {time.time() - start:.0f}s")

    with open(f"results/{name}_metrics.json", "w") as f:
        json.dump({name: {"accuracy": acc, "macro_f1": macro_f1}}, f, indent=2)
    torch.save(
        {"state_dict": best_state, "stoi": stoi, "label_names": label_names,
         "config": {"emb": args.emb, "hidden": args.hidden}},
        f"models/{name}.pt",
    )
    print(f"Saved: results/{name}_metrics.json, models/{name}.pt")


if __name__ == "__main__":
    main()