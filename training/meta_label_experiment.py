"""Meta-labeling tajribasi (deploy modelga TEGMAYDI).

G'oya (López de Prado): modelga "yo'nalish nima?" (Buy/Sell/Hold, 3 sinf) deb emas,
"qoidaga asoslangan birlamchi signal yutadimi?" (2 sinf) deb o'rgatish.

- Birlamchi signal (qoida): trend-following. ADX >= ADX_MIN va SMA20/SMA50/close bir
  yo'nalishda -> Buy yoki Sell. Aks holda signal yo'q.
- Natija (R): TP=+2 ATR, SL=-1 ATR, gorizont HORIZON bar; bir barda ikkalasi ham
  bo'lsa SL; vaqt tugasa -- close bo'yicha R, [-1, +2] oralig'ida.
- Model: P(R > 0 | xususiyatlar) -- faqat birlamchi signal chiqqan barlarda.
- Baholash: walk-forward (expanding), PURGING (train'dan label oynasi test boshiga
  tegadigan qatorlar olib tashlanadi), blok-bootstrap 95% ishonch oralig'i
  (qo'shni qatorlar ustma-ust label oynasiga ega, shuning uchun oddiy bootstrap emas).

Taqqoslash: model filtri bilan tanlangan savdolar o'rtacha R'si vs filtrsiz hamma
birlamchi signallar o'rtacha R'si. 2:1 da break-even win-rate ~33% (spredsiz).

    python training/meta_label_experiment.py [--stride 4] [--rebuild]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "backend"))
import mt5_context  # noqa: E402

RAW_DIR = os.path.join(_THIS_DIR, "..", "data_pipeline", "output", "multi_v2")
CACHE = os.path.join(_THIS_DIR, "..", "data_pipeline", "output", "meta_dataset.parquet")
SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAMES = ["M15", "H1"]
LOOKBACK = 100
HORIZON = 48
ADX_MIN = 20.0
TP_MULT, SL_MULT = mt5_context.TP_ATR_MULT, mt5_context.SL_ATR_MULT


def primary_direction(raw: dict) -> int:
    """+1 Buy, -1 Sell, 0 signal yo'q (trend-following qoidasi)."""
    c, s20, s50 = raw["last_close"], raw["sma_20"], raw["sma_50"]
    if raw["adx_14"] < ADX_MIN:
        return 0
    if s20 > s50 and c > s20:
        return 1
    if s20 < s50 and c < s20:
        return -1
    return 0


def forward_r(direction: int, entry: float, atr: float, highs, lows, close_end: float) -> float:
    tp = entry + direction * TP_MULT * atr
    sl = entry - direction * SL_MULT * atr
    for h, l in zip(highs, lows):
        hit_sl = l <= sl if direction == 1 else h >= sl
        hit_tp = h >= tp if direction == 1 else l <= tp
        if hit_sl:
            return -float(SL_MULT)
        if hit_tp:
            return float(TP_MULT)
    r = (close_end - entry) * direction / atr
    return float(np.clip(r, -SL_MULT, TP_MULT))


def build_rows(args) -> list[dict]:
    symbol, tf, stride = args
    raw = pd.read_parquet(os.path.join(RAW_DIR, f"{symbol}_{tf}_raw.parquet"))
    H, L, C = (raw[k].to_numpy(dtype=np.float64) for k in ("high", "low", "close"))
    times = pd.to_datetime(raw["time"]).to_numpy()
    rows = []
    for idx in range(LOOKBACK, len(raw) - HORIZON - 1, stride):
        s = idx - LOOKBACK + 1
        ind = mt5_context.compute_raw_indicators(
            H[s:idx + 1], L[s:idx + 1], C[s:idx + 1], bar_hour=pd.Timestamp(times[idx]).hour
        )
        d = primary_direction(ind)
        if d == 0 or not ind["atr_14"]:
            continue
        r = forward_r(d, C[idx], ind["atr_14"], H[idx + 1: idx + 1 + HORIZON], L[idx + 1: idx + 1 + HORIZON],
                      C[idx + HORIZON])
        feats = dict(zip(mt5_context.FEATURE_NAMES, mt5_context.to_feature_vector(ind)))
        rows.append({**feats, "dir": d, "is_h1": int(tf == "H1"), "symbol": symbol, "tf": tf,
                     "time": times[idx], "t_end": times[idx + HORIZON], "R": r})
    return rows


def load_dataset(stride: int, rebuild: bool) -> pd.DataFrame:
    if os.path.exists(CACHE) and not rebuild:
        return pd.read_parquet(CACHE)
    jobs = [(s, tf, stride) for s in SYMBOLS for tf in TIMEFRAMES]
    print(f"Dataset quriladi ({len(jobs)} fayl, stride={stride})...", flush=True)
    with ProcessPoolExecutor() as ex:
        all_rows = [r for chunk in ex.map(build_rows, jobs) for r in chunk]
    df = pd.DataFrame(all_rows).sort_values("time").reset_index(drop=True)
    df.to_parquet(CACHE)
    return df


def block_bootstrap_ci(values: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Ketma-ket (vaqt bo'yicha tartiblangan) qiymatlar uchun o'rtacha'ning 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < block * 2:
        return (float("nan"), float("nan"))
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        sample = np.concatenate([values[s:s + block] for s in starts])[:n]
        means[b] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def make_model():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=5.0,
        min_samples_leaf=200, random_state=42,
    )


def walk_forward(df: pd.DataFrame, n_folds: int = 5, first_train_frac: float = 0.5):
    """Expanding walk-forward. Qaytaradi: test qatorlari + model ehtimoli (p)."""
    feature_cols = mt5_context.FEATURE_NAMES + ["dir", "is_h1"]
    times = df["time"]
    t0, t1 = times.min(), times.max()
    edges = [t0 + (t1 - t0) * (first_train_frac + (1 - first_train_frac) * i / n_folds) for i in range(n_folds + 1)]
    parts = []
    for i in range(n_folds):
        test_start, test_end = edges[i], edges[i + 1]
        test = df[(times >= test_start) & (times < test_end)]
        # purging: label oynasi test boshiga yetadigan train qatorlar olib tashlanadi
        train = df[(df["t_end"] < test_start)]
        if len(test) < 200 or len(train) < 1000:
            continue
        model = make_model().fit(train[feature_cols], (train["R"] > 0).astype(int))
        p = model.predict_proba(test[feature_cols])[:, 1]
        parts.append(test.assign(p=p, fold=i))
    return pd.concat(parts)


def report(name: str, sub: pd.DataFrame, block: int):
    if len(sub) == 0:
        print(f"{name:<34} n=0")
        return
    sub = sub.sort_values("time")
    r = sub["R"].to_numpy()
    lo, hi = block_bootstrap_ci(r, block)
    print(f"{name:<34} n={len(sub):>6}  win={(r > 0).mean():.1%}  meanR={r.mean():+.3f}  95%CI[{lo:+.3f}, {hi:+.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    df = load_dataset(args.stride, args.rebuild)
    block = max(2, HORIZON // args.stride)  # ustma-ust label oynasini qamrab oladigan blok uzunligi
    print(f"Dataset: {len(df)} birlamchi signal, {df['time'].min()} .. {df['time'].max()}")
    print(f"Hamma birlamchi signallar R: mean={df['R'].mean():+.3f}, win={(df['R'] > 0).mean():.1%}\n")

    res = walk_forward(df)
    print(f"Walk-forward test qatorlari: {len(res)} (folds: {sorted(res['fold'].unique())})\n")
    aucs = [roc_auc_score((g["R"] > 0).astype(int), g["p"]) for _, g in res.groupby("fold")]
    print(f"Saralash qobiliyati (AUC, 0.5 = tasodifiy): har fold {[round(a, 3) for a in aucs]}, o'rtacha {np.mean(aucs):.3f}\n")
    report("Filtrsiz (hamma birlamchi signal)", res, block)
    for thr in (0.40, 0.45, 0.50):
        report(f"Model filtri p >= {thr:.2f}", res[res["p"] >= thr], block)
    for q in (0.5, 0.3):
        # har fold ichida eng yuqori p'li q ulush -- ehtimol kalibrlanishiga bog'liq emas
        cutoff = res.groupby("fold")["p"].transform(lambda s: s.quantile(1 - q))
        top = res[res["p"] >= cutoff]
        report(f"Model filtri: har fold top {q:.0%}", top, block)

    print("\nSymbol/timeframe bo'yicha (p >= 0.45 vs filtrsiz):")
    for (sym, tf), g in res.groupby(["symbol", "tf"]):
        report(f"  {sym} {tf} filtrsiz", g, block)
        report(f"  {sym} {tf} p>=0.45", g[g["p"] >= 0.45], block)
    print("\nEslatma: 2:1 da break-even win-rate ~33% (spred/slippagesiz). CI nol'ni o'z ichiga olsa -- ustunlik isbotlanmagan.")


if __name__ == "__main__":
    main()
