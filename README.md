# Grid-Bot — Trading Chart Analysis Engine

> 🇺🇿 O'zbekcha hujjat: [README.uz.md](README.uz.md)
> ⚠️ **Not financial advice.** The signal is a decision-support tool only. Always
> use your own risk management and test on a demo account before trading real money.

Upload a candlestick chart screenshot. The system runs **two independent
analysts** over it and merges their verdicts:

1. **Numeric model** — a scikit-learn model trained on technical indicators
   (RSI, MACD, ATR, SMA, Bollinger %B) pulled live from MetaTrader 5. It never
   sees the image.
2. **Gemini vision** — Google's multimodal model looks at the chart *and* the
   same live indicator context, and reasons about trend, support/resistance and
   candle patterns.

When both agree and the numeric model passes its confidence threshold, a
directional signal can be given. Gemini can also override an opposing numeric
signal at 65% confidence, or provide direction when the numeric model says Hold.
When MT5 context is unavailable, the final signal is Hold. Every run is stored
and shown on a history dashboard; model confidence is not a calibrated win probability.

## Why it's built this way

Two uncorrelated sources of evidence (pure numbers vs. visual pattern reading)
catch each other's blind spots. The consensus gate trades signal frequency for
signal quality — see [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) for the
approaches that were tried and rejected, and [`TESTING_LOG.md`](TESTING_LOG.md)
for the running trial log.

## Stack

| Layer | Tech |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn, SQLAlchemy + SQLite |
| ML | scikit-learn, a PyTorch sequence model, joblib |
| Market data | MetaTrader5 Python API (live price + indicators) |
| Vision | `google-genai` (Gemini) |
| Frontend | Vanilla HTML/CSS/JS dashboard, served by the backend |
| Data pipeline | pandas, pyarrow, mplfinance for history fetch + chart rendering |

## Architecture

```
chart image + symbol/timeframe
        │
        ▼
┌───────────────────┐     ┌──────────────────────┐
│  mt5_context.py   │────▶│  numeric model       │  (indicators only)
│  live RSI/MACD/…  │     │  custom_model.py     │──┐
└───────────────────┘     └──────────────────────┘  │
        │                                           ▼
        │                 ┌──────────────────────┐  consensus.py
        └────────────────▶│  gemini_vision.py    │──▶  Buy/Sell/Hold
              image +     │  (image + context)   │     + Entry/TP/SL
              context     └──────────────────────┘     + rationale
                                                        │
                                                        ▼
                                              SQLite history → dashboard
```

## Run it

Requirements: Windows + MetaTrader 5 terminal (installed, logged into a
demo/real account, kept open), Python 3.11+, and a free Gemini API key from
[aistudio.google.com](https://aistudio.google.com).

```bash
git clone https://github.com/mirazizGG/grid-bot.git
cd grid-bot

python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r backend/requirements.txt

copy backend\.env.example backend\.env   # then put your GEMINI_API_KEY in it

cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8811
```

Open **http://127.0.0.1:8811**. MT5 running and logged in is recommended (for
live indicators). Without MT5, Gemini can still analyze the image, but the final
signal is Hold. Outcome evaluation uses ticks for the first partial minute and
complete UTC M1 candles afterwards; gaps are reported as missing data. Costs are not fully modeled.
Older evaluation results remain in history but are excluded from the new statistics.

### Moving history between machines

`backend/analyses.db` and `backend/uploads/` are not in git. To carry your
analysis history over:

```bash
python backend/db_backup.py export            # -> backups/grid-bot-backup-<time>.zip
python backend/db_backup.py import <zip>      # merges; never deletes, skips duplicates
```

## Repo layout

```
backend/         FastAPI server: analysis, consensus, history
  mt5_context.py     live price + RSI/MACD/ATR/SMA/Bollinger from MT5
  custom_model.py    active numeric-indicator model
  sequence_model.py  PyTorch sequence model
  gemini_vision.py   Gemini image + context analysis
  consensus.py       merge logic for the two verdicts
  reasoning.py       human-readable rationale builder
  db.py              SQLite history
frontend/        dashboard (served by the backend)
data_pipeline/   MT5 history fetch → label generation → chart rendering
training/        model training scripts (numeric + Colab CNN + sequence)
```

Large / regenerable assets (historical OHLC, chart image datasets, the legacy
CNN model) are gitignored — the table in [`README.uz.md`](README.uz.md#muhim-eslatma-katta-fayllar-repoda-yoq)
lists how to regenerate each. The active model
(`backend/model/numeric_model.joblib`, ~750 KB) **is** committed, so no
retraining is needed to run the app.

## License

MIT — see [LICENSE](LICENSE).
