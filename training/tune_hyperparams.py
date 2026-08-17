"""GradientBoostingClassifier giperparametrlarini Optuna bilan qidiradi.

train_numeric_model.py bilan bir xil dataset/feature pipeline'dan foydalanadi
(train/serve parity buzilmasligi uchun). Faqat model sozlamalarini
optimallashtiradi -- feature yoki data o'zgarmaydi.

VAL to'plamda umumiy accuracy'ni maksimallashtiradi. Eng yaxshi topilgan
parametrlar bilan TEST natijasini ko'rsatadi (symbol/timeframe kesimida ham),
lekin modelni **saqlamaydi** -- yakuniy qarorni train_numeric_model.py orqali
qo'lda qabul qilish uchun.
"""
import os
import sys

import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "backend"))
sys.path.insert(0, _THIS_DIR)

import mt5_context  # noqa: E402
from train_numeric_model import COMBINED_LABELED, build_dataset  # noqa: E402

import pandas as pd  # noqa: E402


def main():
    labeled = pd.read_parquet(COMBINED_LABELED)
    print("Feature'lar hisoblanmoqda...")
    dataset = build_dataset(labeled)
    print(f"{len(dataset)} qator tayyor")

    X = dataset[mt5_context.FEATURE_NAMES]
    y = dataset["label"]
    train_mask = dataset["split"] == "train"
    val_mask = dataset["split"] == "val"
    test_mask = dataset["split"] == "test"

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "random_state": 42,
        }
        model = GradientBoostingClassifier(**params)
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[val_mask])
        return accuracy_score(y[val_mask], preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150, show_progress_bar=False)

    print("\n=== Eng yaxshi parametrlar (VAL accuracy asosida) ===")
    print(f"VAL accuracy: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_params = dict(study.best_params)
    best_params["random_state"] = 42
    final_model = GradientBoostingClassifier(**best_params)
    final_model.fit(X[train_mask], y[train_mask])

    for name, mask in [("VAL", val_mask), ("TEST", test_mask)]:
        preds = final_model.predict(X[mask])
        acc = accuracy_score(y[mask], preds)
        print(f"\n=== {name} (accuracy={acc:.3f}) ===")
        print(classification_report(y[mask], preds, target_names=final_model.classes_))
        print(confusion_matrix(y[mask], preds, labels=final_model.classes_))

    print("\n=== TEST symbol/timeframe kesimida ===")
    test_df = dataset[test_mask].copy()
    test_df["pred"] = final_model.predict(X[test_mask])
    for (symbol, timeframe), grp in test_df.groupby(["symbol", "timeframe"]):
        acc = accuracy_score(grp["label"], grp["pred"])
        print(f"  {symbol} {timeframe}: n={len(grp)} accuracy={acc:.3f}")

    print("\n(Eslatma: bu skript modelni saqlamaydi. Agar natija joriy 46%")
    print(" (XAUUSD M15 TEST) dan yaxshi bo'lsa, train_numeric_model.py dagi")
    print(" GradientBoostingClassifier parametrlarini shu qiymatlarga o'zgartirib")
    print(" qayta ishga tushiring.)")


if __name__ == "__main__":
    main()
