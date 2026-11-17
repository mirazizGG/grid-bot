# Grid-Bot — Trading Chart Analysis Engine

> 🇺🇿 O'zbekcha hujjat: [README.uz.md](README.uz.md)
> ⚠️ **Not financial advice.** The signal is a decision-support tool only. Always
> use your own risk management and test on a demo account before trading real money.

Upload a candlestick chart screenshot. The system combines two model results:

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

The models use different inputs, but both receive the same MT5 indicators, so
their agreement does not establish independent confirmation or profitability.
See [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) for the
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

### Windows + Docker Desktop

Use Docker Desktop with Linux containers/WSL 2, Python 3.11 x64, and a
logged-in Windows MT5 terminal. From the extracted project folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-docker.ps1
```

Open **http://127.0.0.1:8811** and enter the Gemini key in Settings.
Subsequent sessions use `scripts/start-docker.ps1`; shutdown uses
`scripts/stop-docker.ps1` with the same PowerShell flags. These flags apply
to that process only and do not change the machine execution policy.

The scripts install a small Windows host environment, generate a local token,
start a hidden connector and build/start the Linux application. They refuse to
terminate unrelated processes. Persistent database, uploads, settings, models
and backups live in `.docker-state/`. Stopping/recreating a container preserves
that directory. Start the helper again after Windows sign-in; MT5 must stay open.
Optional Ollama/Qwen remains on Windows and is controlled through the helper.

The connector exposes no TCP listener: bounded, authenticated JSON file-spool
RPC crosses the shared directory. Only read-only MT5 operations and narrowly
scoped host controls are supported. No order API, arbitrary command, privileged
container or Docker socket mount is exposed. The published UI port binds only
to `127.0.0.1`; loopback Host/origin validation remains active inside Docker.
Telemetry identifies Windows-host resources separately from the container API
process. Missing helper/data returns unavailable status rather than fake values.

Set `$env:GRID_BOT_HTTP_PORT = '8812'` before a script to select another local
port. Use `docker compose ps` and `docker compose logs --tail 80 app` to diagnose.
Candidate artifacts/datasets are optional; the active model is included.

### Native Windows

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
pythonw start_server.pyw
```

Open **http://127.0.0.1:8811**. MT5 running and logged in is recommended (for
live indicators). Without MT5, Gemini can still analyze the image, but the final
signal is Hold. Outcome evaluation uses historical Bid/Ask ticks throughout;
visible gaps over 60 seconds are reported as missing data. Broker tick delivery
cannot be proven complete, and commission and slippage are not modeled.
Older evaluation results remain in history but are excluded from the new statistics.

The landing control panel monitors CPU, RAM, disk, process uptime and available
NVIDIA GPUs. It manages analysis pause, provider enablement, write-only Gemini
credentials, model names, confidence filters, weights and analysis defaults.
Settings persist atomically under `backend/runtime/` and apply to new requests.
Models are validated and backed up before explicit activation; rollback and
interrupted-swap recovery preserve the previous artifact/metadata pair.

Controls also provide background evaluation/validation/backup jobs, filtered
history with pagination, CSV export, ZIP downloads and redacted bounded logs.
Ollama stop/restart affects only a process launched by this app. Backend restart
requires the supervisor (`start_server.pyw`) and waits for other jobs to finish;
direct Uvicorn startup remains supported without that restart capability.
Evaluation plans freeze a 24-hour horizon and UTC+0; other `/evaluate` request
parameters are rejected. Candle-cache timestamps are checked as well as quotes,
with a conservative seven-day allowance for weekend/holiday gaps.

The server and history can open without a Gemini key; analysis reports the missing
configuration. Gemini responses are validated before persistence. Optional Qwen
analysis starts through `POST /history/{id}/qwen` after the primary result is
published. Its hypothetical plan freezes its own completion-time quote and timestamp,
so Gemini/Qwen statistics can have different entry times and sample populations.

The API accepts loopback clients and localhost/127.0.0.1/::1 hostnames only.
Browser requests must be same-origin; all POST requests must include
`X-Grid-Bot-Request: 1`. Uploads are
limited to 10 MiB and 25 million pixels. An old file timestamp forces Hold and
excludes the analysis from signal statistics. File timestamps do not prove when
the chart was captured; verify the chart before analysis.

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

The active model is explicitly marked `legacy_unverified`: its artifact hash is
known, but its original training lineage cannot be reconstructed. To rebuild labels
with shared chronological splits and backups, run
`python data_pipeline/relabel_from_raw.py --variant all`, then
`python training/train_numeric_model.py`. Training writes a candidate with data/model
hashes and validation metrics to `training/model_candidate/`; it does not replace
the active model. Reused historical periods do not constitute a new untouched holdout.

Run isolated regression checks with `python -m pytest backend training/test_research.py -q`
and `node --test frontend/test_app.cjs`; neither calls live brokers or paid APIs.

## License

MIT — see [LICENSE](LICENSE).
