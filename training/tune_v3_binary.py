"""2-klass (Buy/Sell) qayta formatlash tajribasi.

G'oya: Hold label'i statistik jihatdan eng "shovqinli" klass (narx kam
harakatlanganda ATR bo'yicha TP ham SL ham urilmaydi -- bu ko'p turli xil
sabablardan bo'lishi mumkin, model uchun aniq pattern topish qiyin). Agar
model faqat Buy vs Sell farqini o'rgansa (Hold qatorlarini o'qitishdan olib
tashlab), qaror chegarasi aniqroq bo'lishi mumkin. Inference paytida esa
predict_proba() eng yuqori ehtimoli past bo'lsa (ya'ni ikkala klass ham
unchalik ishonch bilan aytilmasa) -- Hold deb belgilanadi.

Baseline bilan solishtirish uchun: xuddi shu TEST qatorlarida (Hold
qatorlari ham kiradi!) baholanadi -- ya'ni "modelning umumiy 3 klassni
farqlash qobiliyati" xolisona solishtiriladi.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "backend"))
sys.path.insert(0, _THIS_DIR)

import mt5_context  # noqa: E402
from train_numeric_model import COMBINED_LABELED, build_dataset  # noqa: E402


def main():
    labeled = pd.read_parquet(COMBINED_LABELED)
    dataset = build_dataset(labeled)
    X_cols = mt5_context.FEATURE_NAMES

    train_mask = dataset["split"] == "train"
    test_mask = dataset["split"] == "test"

    # faqat Buy/Sell qatorlarida o'qitish
    bs_train_mask = train_mask & dataset["label"].isin(["Buy", "Sell"])
    X_train_bs = dataset[bs_train_mask][X_cols]
    y_train_bs = dataset[bs_train_mask]["label"]
    print(f"Binary train set: {len(X_train_bs)} qator (Hold chiqarib tashlandi)")

    model = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(X_train_bs, y_train_bs)

    # Hold uchun eshik: eng yuqori class ehtimoli shu qiymatdan past bo'lsa -> Hold
    X_test = dataset[test_mask][X_cols]
    y_test = dataset[test_mask]["label"]
    proba = model.predict_proba(X_test)  # [P(Buy), P(Sell)]
    max_proba = proba.max(axis=1)
    raw_pred = model.classes_[proba.argmax(axis=1)]

    for hold_threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        pred = np.where(max_proba >= hold_threshold, raw_pred, "Hold")
        acc = accuracy_score(y_test, pred)

        sub = dataset[test_mask].copy()
        sub["pred"] = pred
        xau_m15 = sub[(sub["symbol"] == "XAUUSD") & (sub["timeframe"] == "M15")]
        xau_acc = accuracy_score(xau_m15["label"], xau_m15["pred"])
        hold_frac = (pred == "Hold").mean()
        print(f"threshold={hold_threshold:.2f}  TEST overall={acc:.3f}  XAUUSD M15={xau_acc:.3f}  Hold ulushi={hold_frac:.2f}")


if __name__ == "__main__":
    main()
