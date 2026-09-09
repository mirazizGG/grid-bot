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

When both agree → a higher-confidence **Buy / Sell / Hold** signal with
ATR-based Entry / TP / SL levels and a written rationale. When they disagree →
**Hold**, by design. Every run is stored and shown on a history dashboard.

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
live indicators) but not required — without it Gemini falls back to the image
alone.

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
