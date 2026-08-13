"""Har bir label'langan window uchun candlestick PNG rasm yaratadi va manifest.csv yozadi.

Rasmlar indikatorsiz, sodda holatda chiziladi (real skrinshotlarga o'xshash bo'lishi uchun).
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import mplfinance as mpf
import pandas as pd

from config import RAW_DATA_PATH, LABELED_DATA_PATH, DATASET_DIR

FIGSIZE = (4, 4)
DPI = 100

MPF_STYLE = mpf.make_mpf_style(
    base_mpf_style="charles",
    rc={"axes.grid": False},
)


def render_one(ohlc_window: pd.DataFrame, out_path: str) -> None:
    fig, _ = mpf.plot(
        ohlc_window.set_index("time"),
        type="candle",
        style=MPF_STYLE,
        figsize=FIGSIZE,
        axisoff=True,
        returnfig=True,
        tight_layout=True,
    )
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    matplotlib.pyplot.close(fig)


def main(limit: int | None):
    raw = pd.read_parquet(RAW_DATA_PATH)
    labeled = pd.read_parquet(LABELED_DATA_PATH)

    if limit:
        samples = [
            g.sample(min(len(g), limit), random_state=42)
            for _, g in labeled.groupby(["split", "label"])
        ]
        labeled = pd.concat(samples, ignore_index=True)

    manifest_rows = []
    total = len(labeled)
    for n, (_, row) in enumerate(labeled.iterrows(), start=1):
        split, label = row["split"], row["label"]
        out_dir = os.path.join(DATASET_DIR, "images", split, label)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{int(row['window_end_idx'])}.png")

        if not os.path.exists(out_path):
            window = raw.iloc[int(row["window_start_idx"]): int(row["window_end_idx"]) + 1]
            render_one(window, out_path)

        manifest_rows.append(
            {
                "image_path": out_path.replace("\\", "/"),
                "split": split,
                "label": label,
                "entry_price": row["entry_price"],
                "tp_price": row["tp_price"],
                "sl_price": row["sl_price"],
                "time": row["time"],
            }
        )
        if n % 500 == 0 or n == total:
            print(f"{n}/{total} rasm tayyor")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(DATASET_DIR, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"Manifest saqlandi -> {manifest_path} ({len(manifest)} qator)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit-per-group",
        type=int,
        default=None,
        help="Har bir (split,label) guruhi uchun maksimal rasm soni (test/tezkor ishga tushirish uchun)",
    )
    args = parser.parse_args()
    main(args.limit_per_group)
