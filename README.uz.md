# Grid-Bot — MT5 Chart Tahlil Tizimi

Savdo chart (candlestick) skrinshotini yuklaysiz — tizim ikkita mustaqil
manbani solishtirib (o'zi o'qitilgan raqamli model + Gemini AI vision),
yakuniy **Buy/Sell/Hold** signalini, Entry/TP/SL narxlarini va nega shunday
qaror qabul qilinganini chiqaradi. Natijalar tarix sifatida saqlanadi va
Dashboard'da ko'riladi.

> **Loyihani birinchi marta ko'rayotgan bo'lsangiz (yoki Claude Code'ga
> "o'rganib chiq" desangiz): avval [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md)
> ni o'qing** — u yerda nima qilingani, nega hozirgi qarorlar qabul
> qilingani va qaysi yo'llar sinab ko'rilib rad etilgani yozilgan.
> Joriy 1-haftalik sinov jurnali: [`TESTING_LOG.md`](TESTING_LOG.md).

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

MT5 terminali ochiq va hisobga kirilgan bo'lishi kerak. U ulanmagan bo'lsa Gemini rasmni tahlil qiladi, lekin yakuniy signal Hold bo'ladi.

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
5. Ikkalasi kelishsa va custom ishonchi yetarli bo'lsa signal beriladi. Qarama-qarshi
   yo'nalishda Gemini bahosi 65% yoki yuqori bo'lsa, Gemini yo'nalishi olinadi;
   custom Hold bo'lsa ham Gemini yo'nalishi olinishi mumkin. MT5 konteksti yo'q
   bo'lsa yakuniy qaror **Hold**. TP/SL joriy MT5 narxi va ATR asosida hisoblanadi.
6. Natija tarixga saqlanadi, Dashboard'da ko'rinadi.

Qo'shimcha ravishda Ollama orqali mahalliy **Qwen 3.5 4B** rasm va ayni MT5
kontekstini tahlil qiladi. Uning xulosasi taqqoslash uchun va yakuniy signalga
ta'sir qilmaydi. Ollama ishlayotgan va `ollama pull qwen3.5:4b` bajarilgan
bo'lishi kerak. Model ishlamasa, uning xatosi ko'rsatiladi va Gemini tahlili
davom etadi. `.env` orqali `QWEN_MODEL`, `OLLAMA_URL`, `QWEN_TIMEOUT` sozlanadi.

Dashboarddagi Gemini/Qwen natijalari alohida **gipotetik** taqqoslashdir: har
bir yo'nalish uchun bir xil saqlangan MT5 narxi, ATR va 24 soatlik muddat
ishlatiladi. Qwen faqat yo'nalish beradi, shuning uchun unda darhol kirish
faraz qilinadi; Gemini kutishni aytganda uning rejasi hisoblanmaydi. Ochiq,
ma'lumotsiz va baholanmagan holatlar TP ulushiga kirmaydi. Modellar namunalari
soni farq qilishi mumkin. Eski tahlillar qayta tahlil qilinmaydi va ular uchun
yangi gipotetik rejalar to'qib yaratilmaydi.

Baholash odatda 24 soat davomida birinchi qisman daqiqadagi ticklarni va keyingi
to'liq 1 daqiqalik shamlarni tekshiradi. Tick yoki bar tarixida bo'shliq bo'lsa
natija ma'lumotsiz qolishi mumkin. Spread va komissiya to'liq hisoblanmaydi.
Eski usulda baholangan natijalar tarixda saqlanadi, yangi statistikaga qo'shilmaydi.
Model bahosi haqiqiy savdo muvaffaqiyati ehtimoli sifatida tasdiqlanmagan.

Tahlil oxirida narx yangiligi qayta tekshiriladi. Kotirovka 60 soniyadan eski,
narx tahlil davomida 0.25 ATR dan ko'p o'zgargan yoki Gemini kirishni kutishni
aytgan bo'lsa Hold qaytariladi. Kutish sharti bajarilganda qayta tahlil qiling.
Windows kirishida avtomatik ishga tushirish faqat foydalanuvchi tizimga kirganda
ishlaydi; MT5 ochiq va hisobga ulangan bo'lishi alohida talab qilinadi.
