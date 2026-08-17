"""Custom modelni kuchaytirish uchun bir nechta yondashuvni ketma-ket sinaydi:

  A. sklearn GradientBoostingClassifier, class_weight muammosi yo'q (u
     class_weight qo'llab-quvvatlamaydi, shuning uchun sample_weight ishlatiladi)
  B. sklearn HistGradientBoostingClassifier (class_weight='balanced')
  C. LightGBM (class_weight='balanced')
  D. XGBoost (sample_weight balanced)
  E. Optuna qidiruv (eng yaxshi chiqqan oila ustida), lekin bu safar
     **XAUUSD M15 VAL accuracy**ga optimallashtiriladi (avvalgi xato --
     umumiy VAL accuracy'ga optimallashtirish Hold klassini yo'q qilib,
     XAUUSD M15'ni yomonlashtirgan edi).

Hech narsani diskka yozmaydi -- faqat solishtirish uchun.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "backend"))
sys.path.insert(0, _THIS_DIR)

import mt5_context  # noqa: E402
from train_numeric_model import COMBINED_LABELED, build_dataset  # noqa: E402

BASELINE_XAUUSD_M15 = 0.46


def xau_m15_acc(model, dataset, mask):
    sub = dataset[mask]
    sub = sub[(sub["symbol"] == "XAUUSD") & (sub["timeframe"] == "M15")]
    if len(sub) == 0:
        return None
    preds = model.predict(sub[mt5_context.FEATURE_NAMES])
    return accuracy_score(sub["label"], preds)


def report(name, model, dataset, val_mask, test_mask):
    val_acc = accuracy_score(dataset[val_mask]["label"], model.predict(dataset[val_mask][mt5_context.FEATURE_NAMES]))
    test_acc = accuracy_score(dataset[test_mask]["label"], model.predict(dataset[test_mask][mt5_context.FEATURE_NAMES]))
    xau_val = xau_m15_acc(model, dataset, val_mask)
    xau_test = xau_m15_acc(model, dataset, test_mask)
    print(f"\n--- {name} ---")
    print(f"  VAL overall={val_acc:.3f}  TEST overall={test_acc:.3f}")
    print(f"  XAUUSD M15: VAL={xau_val:.3f}  TEST={xau_test:.3f}  (baseline TEST={BASELINE_XAUUSD_M15:.2f})")
    return xau_test


def main():
    labeled = pd.read_parquet(COMBINED_LABELED)
    print("Feature'lar hisoblanmoqda...")
    dataset = build_dataset(labeled)
    print(f"{len(dataset)} qator tayyor")

    X_cols = mt5_context.FEATURE_NAMES
    train_mask = dataset["split"] == "train"
    val_mask = dataset["split"] == "val"
    test_mask = dataset["split"] == "test"

    X_train = dataset[train_mask][X_cols]
    y_train = dataset[train_mask]["label"]
    sw_train = compute_sample_weight("balanced", y_train)

    results = {}

    # A. Baseline replica (joriy deployed sozlamalar)
    m = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    m.fit(X_train, y_train)
    results["A_baseline_replica"] = report("A) Joriy deployed sozlama (replika)", m, dataset, val_mask, test_mask)

    # A2. Baseline + balanced sample weight
    m = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    m.fit(X_train, y_train, sample_weight=sw_train)
    results["A2_baseline_balanced"] = report("A2) Joriy sozlama + balanced sample_weight", m, dataset, val_mask, test_mask)

    # B. HistGradientBoosting (balanced)
    m = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, class_weight="balanced", random_state=42)
    m.fit(X_train, y_train)
    results["B_histgbm_balanced"] = report("B) HistGradientBoostingClassifier (balanced)", m, dataset, val_mask, test_mask)

    # B2. HistGradientBoosting (no balance)
    m = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42)
    m.fit(X_train, y_train)
    results["B2_histgbm"] = report("B2) HistGradientBoostingClassifier (balanslanmagan)", m, dataset, val_mask, test_mask)

    # C. LightGBM
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, class_weight="balanced",
                                random_state=42, verbose=-1)
        m.fit(X_train, y_train)
        results["C_lightgbm_balanced"] = report("C) LightGBM (balanced)", m, dataset, val_mask, test_mask)

        m = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
        m.fit(X_train, y_train)
        results["C2_lightgbm"] = report("C2) LightGBM (balanslanmagan)", m, dataset, val_mask, test_mask)
    except ImportError:
        print("LightGBM mavjud emas, o'tkazib yuborildi")

    # D. XGBoost
    try:
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        class XGBWrap:
            def __init__(self, **kw):
                self.model = xgb.XGBClassifier(**kw)
            def fit(self, X, y, sample_weight=None):
                y_enc = le.transform(y)
                self.model.fit(X, y_enc, sample_weight=sample_weight)
                return self
            def predict(self, X):
                return le.inverse_transform(self.model.predict(X))

        m = XGBWrap(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="mlogloss")
        m.fit(X_train, y_train, sample_weight=sw_train)
        results["D_xgboost_balanced"] = report("D) XGBoost (balanced sample_weight)", m, dataset, val_mask, test_mask)

        m = XGBWrap(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="mlogloss")
        m.fit(X_train, y_train)
        results["D2_xgboost"] = report("D2) XGBoost (balanslanmagan)", m, dataset, val_mask, test_mask)
    except ImportError:
        print("XGBoost mavjud emas, o'tkazib yuborildi")

    print("\n\n=== YAKUNIY JADVAL (XAUUSD M15 TEST accuracy) ===")
    print(f"  Baseline (deployed model): {BASELINE_XAUUSD_M15:.3f}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        mark = "  <-- BASELINE'DAN YAXSHI" if acc > BASELINE_XAUUSD_M15 else ""
        print(f"  {name}: {acc:.3f}{mark}")


if __name__ == "__main__":
    main()
