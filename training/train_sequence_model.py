"""LSTM sequence model o'qitadi -- har window uchun oxirgi 30 bar ketma-ketligi
(ATR-normallashtirilgan tana/fitil/diapazon) + joriy statik indikatorlar
(RSI/MACD/ADX va h.k.) asosida Buy/Sell/Hold aytadi.

Faqat XAUUSD (M15+H1), data_pipeline/multi_symbol_dataset.py natijasidan.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import mt5_context  # noqa: E402
import sequence_model  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_DIR = os.path.join(_THIS_DIR, "..", "data_pipeline", "output", "multi")
COMBINED_LABELED = os.path.join(_THIS_DIR, "..", "data_pipeline", "output", "multi_labeled.parquet")

MODEL_OUT = os.path.join(_THIS_DIR, "..", "backend", "model", "sequence_model.pt")
META_OUT = os.path.join(_THIS_DIR, "..", "backend", "model", "sequence_model_meta.json")

LOOKBACK_BARS = 100
MIN_HISTORY = 55
LABEL_MAP = {"Buy": 0, "Sell": 1, "Hold": 2}


def build_dataset(labeled: pd.DataFrame):
    seqs, statics, labels, splits = [], [], [], []

    for (symbol, timeframe), group in labeled.groupby(["symbol", "timeframe"]):
        raw = pd.read_parquet(os.path.join(MULTI_DIR, f"{symbol}_{timeframe}_raw.parquet"))
        opens_all = raw["open"].to_numpy(dtype=np.float64)
        highs_all = raw["high"].to_numpy(dtype=np.float64)
        lows_all = raw["low"].to_numpy(dtype=np.float64)
        closes_all = raw["close"].to_numpy(dtype=np.float64)
        times_all = raw["time"]

        for _, row in group.iterrows():
            idx = int(row["window_end_idx"])
            start = max(0, idx - LOOKBACK_BARS + 1)
            if idx - start < MIN_HISTORY:
                continue

            o, h, l, c = opens_all[start:idx + 1], highs_all[start:idx + 1], lows_all[start:idx + 1], closes_all[start:idx + 1]
            bar_hour = pd.to_datetime(times_all.iloc[idx]).hour

            raw_ind = mt5_context.compute_raw_indicators(h, l, c, bar_hour=bar_hour)
            static_features = mt5_context.to_feature_vector(raw_ind)
            seq = sequence_model.build_bar_sequence(o, h, l, c, raw_ind["atr_14"])

            seqs.append(seq)
            statics.append(static_features)
            labels.append(LABEL_MAP[row["label"]])
            splits.append(row["split"])

    return (np.stack(seqs), np.array(statics, dtype=np.float32),
            np.array(labels, dtype=np.int64), np.array(splits))


class ArrDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, statics, labels):
        self.seqs, self.statics, self.labels = seqs, statics, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.seqs[i], self.statics[i], self.labels[i]


def run_epoch(model, loader, optimizer, criterion, train_mode, device):
    model.train() if train_mode else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for seq, static, y in loader:
            seq, static, y = seq.to(device), static.to(device), y.to(device)
            if train_mode:
                optimizer.zero_grad()
            logits = model(seq, static)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def main():
    labeled = pd.read_parquet(COMBINED_LABELED)
    print("Sequence + static feature'lar hisoblanmoqda...")
    seqs, statics, labels, splits = build_dataset(labeled)
    print(f"{len(labels)} qator tayyor, sequence shape={seqs.shape}, static dim={statics.shape[1]}")

    # static feature'larni normallashtirish (LSTM chiqishi bilan bir shkalada bo'lishi uchun)
    train_mask = splits == "train"
    static_mean = statics[train_mask].mean(axis=0)
    static_std = statics[train_mask].std(axis=0) + 1e-6
    statics_norm = (statics - static_mean) / static_std

    device = torch.device("cpu")
    datasets = {}
    for name, mask in [("train", splits == "train"), ("val", splits == "val"), ("test", splits == "test")]:
        datasets[name] = ArrDataset(
            torch.tensor(seqs[mask]), torch.tensor(statics_norm[mask]), torch.tensor(labels[mask])
        )

    loaders = {
        name: torch.utils.data.DataLoader(ds, batch_size=64, shuffle=(name == "train"))
        for name, ds in datasets.items()
    }

    model = sequence_model.SequenceModel(static_dim=statics.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc, patience_left, EPOCHS, PATIENCE = 0.0, 8, 60, 8
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], optimizer, criterion, True, device)
        val_loss, val_acc = run_epoch(model, loaders["val"], optimizer, criterion, False, device)
        scheduler.step(val_acc)
        print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} acc {train_acc:.3f} | val_loss {val_loss:.4f} acc {val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc, patience_left = val_acc, PATIENCE
            torch.save(model.state_dict(), "_best_seq_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Erta to'xtatildi")
                break

    model.load_state_dict(torch.load("_best_seq_model.pt"))
    model.eval()

    for name in ["val", "test"]:
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seq, static, y in loaders[name]:
                logits = model(seq, static)
                all_preds.extend(logits.argmax(1).numpy())
                all_labels.extend(y.numpy())
        acc = accuracy_score(all_labels, all_preds)
        print(f"\n=== {name.upper()} (accuracy={acc:.3f}) ===")
        print(classification_report(all_labels, all_preds, target_names=["Buy", "Sell", "Hold"]))
        print(confusion_matrix(all_labels, all_preds))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    with open(META_OUT, "w") as f:
        json.dump({
            "static_dim": int(statics.shape[1]),
            "static_mean": static_mean.tolist(),
            "static_std": static_std.tolist(),
            "feature_names": mt5_context.FEATURE_NAMES,
            "classes": ["Buy", "Sell", "Hold"],
        }, f, indent=2)

    os.remove("_best_seq_model.pt")
    print(f"\nSaqlandi -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
