# Grid-Bot — MT5 Chart Tahlil Tizimi

Savdo chart (candlestick) skrinshotini yuklaysiz — tizim ikkita mustaqil
manbani solishtirib (o'zi o'qitilgan raqamli model + Gemini AI vision),
yakuniy **Buy/Sell/Hold** signalini, Entry/TP/SL narxlarini va nega shunday
qaror qabul qilinganini chiqaradi. Natijalar tarix sifatida saqlanadi va
Dashboard'da ko'riladi.

> **Moliyaviy maslahat emas.** Signal — yordamchi vosita, xolos. Doim
> risk-management bilan ishlating, real pul bilan savdo qilishdan oldin
> demo hisobda sinab ko'ring.

## Loyiha tuzilishi

```
grid-bot/
├── backend/          FastAPI server: tahlil, consensus, tarix (SQLite)
│   ├── main.py            API endpoint'lar (/analyze, /history, /stats)
│   ├── mt5_context.py     MT5'dan jonli narx + RSI/MACD/ATR/SMA/Bollinger
│   ├── custom_model.py    Raqamli indikatorlar asosidagi model (aktiv)
│   ├── custom_model_legacy_cnn.py   Eski rasm-CNN model (endi ishlatilmaydi)
│   ├── gemini_vision.py   Gemini API orqali chart+kontekst tahlili
│   ├── consensus.py       Ikkala natijani birlashtirish mantig'i
│   ├── db.py              SQLite (tarix)
│   └── model/             Raqamli model fayli (numeric_model.joblib)
├── frontend/         Vanilla HTML/CSS/JS dashboard (backend shu papkani serve qiladi)
├── data_pipeline/    MT5'dan tarixiy ma'lumot olish, label yaratish, chart rasm generatsiyasi
└── training/         Model o'qitish skriptlari (Colab CNN + local raqamli model)
```

## Talablar

- Windows + **MetaTrader 5** terminali (o'rnatilgan, demo/real hisobga kirilgan, ochiq turishi kerak)
- Python 3.11+
- Gemini API kaliti ([aistudio.google.com](https://aistudio.google.com) dan bepul olinadi)

## O'rnatish (yangi kompyuterda)

```powershell
git clone https://github.com/mirazizGG/grid-bot.git
cd grid-bot

python -m venv .venv
.venv\Scripts\activate

pip install -r backend/requirements.txt
```

### API kalitni sozlash

```powershell
copy backend\.env.example backend\.env
```

`backend\.env` faylini oching va haqiqiy Gemini kalitingizni qo'ying:

```
GEMINI_API_KEY=sizning_kalitingiz
```

### Ishga tushirish

MT5 terminali ochiq va hisobga kirilgan bo'lishi kerak (jonli narx/RSI/MACD olish uchun — bo'lmasa ham ishlaydi, lekin Gemini faqat rasmga qarab taxmin qiladi).

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8811
```

Brauzerda oching: **http://127.0.0.1:8811**

## Muhim eslatma: katta fayllar repo'da yo'q

Quyidagilar `.gitignore` orqali repo'dan chiqarib tashlangan (juda katta va
qayta generatsiya qilinadigan bo'lgani uchun):

| Nima | Qayerda | Qayta yaratish |
|---|---|---|
| Tarixiy OHLC + chart rasmlari | `dataset/`, `data_pipeline/output/` | `data_pipeline/mt5_fetch.py` → `label_generator.py` → `render_charts.py` |
| Eski rasm-CNN model | `backend/model/chart_signal_model.onnx*` | `training/train_colab.ipynb` (Google Colab, GPU) — hozir ishlatilmaydi |
| `.env` (API kalit) | `backend/.env` | Yuqoridagi "API kalitni sozlash" bo'limi |

**`backend/model/numeric_model.joblib`** (aktiv model, ~750 KB) repo'da **bor** —
qayta o'qitish shart emas. Agar qayta o'qitmoqchi bo'lsangiz:
`training/train_numeric_model.py` (avval `data_pipeline` orqali ma'lumot tayyor bo'lishi kerak).

## Qanday ishlaydi (qisqacha)

1. Siz chart rasmini + Symbol/Timeframe tanlaysiz → yuklaysiz.
2. Backend MT5'dan shu symbol uchun jonli narx, RSI, MACD, ATR, SMA, Bollinger %B oladi.
3. **Custom model** — shu raqamli ko'rsatkichlarga qarab (rasmni ko'rmaydi) Buy/Sell/Hold aytadi.
4. **Gemini** — rasmni + shu raqamli ko'rsatkichlarni birga ko'rib, mustaqil xulosa chiqaradi.
5. Ikkalasi kelishsa — yuqori ishonchli signal, TP/SL ATR asosida aniq hisoblanadi.
   Kelishmasa — xavfsizlik uchun **Hold**.
6. Natija tarixga saqlanadi, Dashboard'da ko'rinadi.
