"""Bazadagi (backend/analyses.db) signallarni MT5 tarixi bilan solishtirib,
natijani bazaga yozadi va xulosani chiqaradi. Mantiq backend/evaluator.py da
(dashboard'dagi "Natijalarni yangilash" tugmasi ham shuni ishlatadi).

Ishga tushirish (MT5 terminali ochiq va hisobga kirilgan bo'lishi kerak):
    python training/evaluate_signals.py [--horizon-hours 24] [--utc-offset 3]

--utc-offset faqat server farqi saqlanmagan eski yozuvlar uchun kerak.
"""
import argparse
import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from db import SessionLocal  # noqa: E402
import evaluator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-hours", type=float, default=evaluator.DEFAULT_HORIZON_HOURS)
    ap.add_argument("--utc-offset", type=int, default=None, help="broker server vaqtining UTC'dan farqi (soat)")
    args = ap.parse_args()

    try:
        res = evaluator.evaluate_pending(args.horizon_hours, args.utc_offset)
    except RuntimeError as exc:
        sys.exit(str(exc))
    print(f"Tekshirildi: {res['checked']}  {res['by_outcome']}\n")

    session = SessionLocal()
    try:
        summary = evaluator.summarize(session)
    finally:
        session.close()

    for kind, s in summary.items():
        if s["n"] == 0:
            print(f"{kind:<16} n=0 (OPEN {s['open']}, EXPIRED {s['expired']}, NO_DATA {s['no_data']})")
            continue
        print(f"{kind:<16} n={s['n']}  TP={s['tp']}  SL={s['sl']}  win-rate={s['win_rate']:.0%}  "
              f"jami={s['total_r']:+.1f}R  o'rtacha={s['avg_r']:+.2f}R  "
              f"(OPEN {s['open']}, EXPIRED {s['expired']}, NO_DATA {s['no_data']})")
    print("\nEslatma: namuna kichik bo'lsa bu raqamlar statistik xulosa emas, faqat yo'nalish.")


if __name__ == "__main__":
    main()
