"""Mavjud (endi tarixiy ma'lumot bilan kengaytirilgan) raw parquet fayllarni
MT5'ga qayta ulanmasdan qayta label'laydi -- multi_symbol_dataset.py'ning
label_windows() funksiyasidan foydalanadi."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import pandas as pd

try:
    from .multi_symbol_dataset import label_windows, assign_shared_splits, OUT_DIR, COMBINED_OUT
except ImportError:
    from multi_symbol_dataset import label_windows, assign_shared_splits, OUT_DIR, COMBINED_OUT

SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAMES = ["M15", "H1"]


def relabel_dataset(out_dir, combined_out, backup_dir):
    """Build completely before backing up and replacing generated labels."""
    all_labeled = []
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            raw_path = os.path.join(out_dir, f"{symbol}_{timeframe}_raw.parquet")
            raw = pd.read_parquet(raw_path)
            print(f"--- {symbol} {timeframe}: {len(raw)} bar ---")

            labeled = label_windows(raw, symbol, timeframe)
            print(f"  {len(labeled)} label'langan window, {labeled['label'].value_counts().to_dict()}")
            all_labeled.append(labeled)

    combined = assign_shared_splits(pd.concat(all_labeled, ignore_index=True))
    outputs = [(Path(combined_out), combined)]
    outputs.extend((Path(out_dir) / f"{symbol}_{timeframe}_labeled.parquet", group)
                   for (symbol, timeframe), group in combined.groupby(["symbol", "timeframe"]))
    backup = Path(backup_dir) / Path(out_dir).name
    backup.mkdir(parents=True, exist_ok=False)
    for target, _ in outputs:
        if target.exists():
            shutil.copy2(target, backup / target.name)
    for target, frame in outputs:
        temporary = target.with_suffix(".parquet.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"\nJami: {len(combined)} qator -> {combined_out}; backup: {backup}")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("multi", "multi_v2", "all"), default="multi")
    default_backup = Path(__file__).resolve().parents[1] / "backups" / ("relabel-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--backup-dir", default=str(default_backup))
    args = parser.parse_args()
    variants = ("multi", "multi_v2") if args.variant == "all" else (args.variant,)
    for variant in variants:
        out_dir = Path(OUT_DIR).parent / variant
        combined_out = Path(COMBINED_OUT).parent / f"{variant}_labeled.parquet"
        relabel_dataset(out_dir, combined_out, args.backup_dir)


if __name__ == "__main__":
    main()
