# Grid-Bot — MT5 Chart Tahlil Tizimi

Savdo chart (candlestick) skrinshotini yuklaysiz — tizim ikkita
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

### Docker orqali Windows'da ishga tushirish

Docker Desktop (Linux containers / WSL 2), Python 3.11 x64 va ochiq,
hisobga kirilgan MT5 terminali kerak. Backend, dashboard, Gemini va raqamli
model Docker'da ishlaydi. Windows'dagi yordamchi faqat MT5 ma'lumotlari,
host resurslari va chegaralangan Ollama amallarini yetkazadi.

PowerShell'da loyiha papkasidan birinchi marta:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-docker.ps1
```

Skript kichik `.venv-host` muhitini o'rnatadi, maxfiy ulanish tokenini
yaratadi, Windows yordamchisini yashirin ishga tushiradi va Docker image'ini
yig'ib ochadi. Dashboard: **http://127.0.0.1:8811**. Gemini kalitini
dashboarddagi Sozlamalarda kiriting. Qwen kerak bo'lsa, Ollama va
`qwen3.5:4b` Windows'da alohida o'rnatilgan bo'lsin.

Keyingi ishga tushirish va to'xtatish:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-docker.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop-docker.ps1
```

`Bypass` faqat shu PowerShell jarayoni uchun; tizimning execution policy'si
o'zgartirilmaydi. To'xtatish ma'lumotlarni o'chirmaydi. Windows qayta
yoqilgandan keyin MT5'ni ochib, start skriptini yana bajaring.

Sozlamalar, baza, rasmlar, modellar va zaxiralar `.docker-state/` ichida
saqlanadi. Konteyner qayta yaratilganda ular saqlanib qoladi. Birinchi
ochilishda faol model image'dan nusxalanadi; mavjud model ustiga yozilmaydi.
Candidate/dataset bu paketda bo'lmasa, model bo'limi uni mavjud emas deb
ko'rsatadi. Faol model ishlashi uchun dataset talab qilinmaydi.

Windows va Docker o'rtasida tarmoq porti ochilmaydi: umumiy papkadagi
chegaralangan JSON xabarlar HMAC bilan imzolanadi. Ko'prik order yuborish,
ixtiyoriy buyruq yoki ixtiyoriy fayl yo'lini qabul qilmaydi. Dashboard
porti faqat `127.0.0.1`ga ochiq; Docker socket yoki privileged rejim
ishlatilmaydi. CPU/RAM/GPU ko'rsatkichlari Windows'dan, backend jarayoni
ko'rsatkichlari konteynerdan olinadi. Ko'prik uzilsa, host qiymatlari
mavjud emas deb ko'rsatiladi va MT5 signali Hold bo'ladi.

8811 band bo'lsa, boshqa jarayonni to'xtatmasdan port tanlang:

```powershell
$env:GRID_BOT_HTTP_PORT = '8812'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-docker.ps1
```

Jurnal va holat:

```powershell
docker compose ps
docker compose logs --tail 80 app
```

### Docker'siz Windows varianti

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

MT5 terminali ochiq va hisobga kirilgan bo'lishi kerak. U ulanmagan bo'lsa Gemini rasmni tahlil qiladi, lekin yakuniy signal Hold bo'ladi. Gemini kaliti hali sozlanmagan bo'lsa server, tarix va xizmat holati ochiladi; yangi Gemini tahlili xato qaytaradi.

```powershell
cd backend
pythonw start_server.pyw
```

Brauzerda oching: **http://127.0.0.1:8811**

## Boshqaruv markazi

Asosiy sahifada CPU, RAM, disk, backend jarayoni va mavjud NVIDIA GPU'lar
ko'rsatkichlari yangilanadi. Shu yerdan yangi tahlillarni vaqtincha to'xtatish,
Gemini/Qwen'ni yoqish, API kaliti va model nomlarini, confidence chegaralari,
ovoz vaznlari hamda boshlang'ich symbol/timeframe'ni saqlash mumkin.
O'zgarishlar yangi tahlillarga darhol qo'llanadi; davom etayotgan tahlil o'z
boshlang'ich sozlamalari bilan tugaydi. Kalit javoblarda qaytarilmaydi.

Ollama'ni panel ishga tushirsa, uni to'xtatish va qayta boshlash ham mumkin.
Tashqarida ishga tushgan Ollama jarayoniga panel egalik qilmaydi; bunday
jarayonni to'xtatish tugmasi o'chiriladi. MT5 tugmasi ulanishni tekshiradi.
Backend restarti `start_server.pyw` nazoratchisi bilan ishlaydi va boshqa
vazifalar tugaguncha bloklanadi. Oddiy `uvicorn main:app` bilan ishga
tushirish ham mumkin, lekin paneldan backend restarti mavjud bo'lmaydi.

Model bo'limi faol va candidate hash'larini, kelib chiqish holati va validation
natijalarini ko'rsatadi. Faollashtirish alohida tasdiqdan keyin bajariladi:
feature/class, model va dataset hash'lari tekshiriladi, oldingi model
zaxiralanadi. Xatoda yoki uzilgan almashtirishda oldingi juftlik tiklanadi.

Vazifalar bo'limida baholash, model tekshiruvi va zaxira olish holati ko'rinadi.
Navbat cheklangan, bir xil davom etayotgan vazifa takrorlanmaydi; server
uzilishidan qolgan vazifalar qayta boshlash uchun xato holatiga o'tadi.
Tarixni filterlash, sahifalash va CSV yuklash, baza/rasmlarni ZIP zaxiralash,
hamda maxfiy qiymatlari yashirilgan loglarni ko'rish ham mavjud.
Sozlamalar va vazifalar `backend/runtime/` ichida, model rollback nusxalari
`backups/model-control/` ichida saqlanadi; ular Git'ga qo'shilmaydi.

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
4. **Gemini** — rasmni + shu raqamli ko'rsatkichlarni birga ko'rib, xulosa chiqaradi.
5. Ikkalasi kelishsa va custom ishonchi yetarli bo'lsa signal beriladi. Qarama-qarshi
   yo'nalishda Gemini bahosi 65% yoki yuqori bo'lsa, Gemini yo'nalishi olinadi;
   custom Hold bo'lsa ham Gemini yo'nalishi olinishi mumkin. MT5 konteksti yo'q
   bo'lsa yakuniy qaror **Hold**. TP/SL joriy MT5 narxi va ATR asosida hisoblanadi.
6. Natija tarixga saqlanadi, Dashboard'da ko'rinadi.

Qo'shimcha ravishda Ollama orqali mahalliy **Qwen 3.5 4B** rasm va ayni MT5
kontekstini tahlil qiladi. Uning xulosasi taqqoslash uchun va yakuniy signalga
ta'sir qilmaydi. Asosiy signal darhol ko'rsatiladi, frontend keyin
`POST /history/{id}/qwen` orqali Qwen'ni alohida chaqiradi. Ollama ishlayotgan va `ollama pull qwen3.5:4b` bajarilgan
bo'lishi kerak. Model ishlamasa, uning xatosi ko'rsatiladi va Gemini tahlili
davom etadi. `.env` orqali `QWEN_MODEL`, `OLLAMA_URL`, `QWEN_TIMEOUT` sozlanadi.

Dashboarddagi Gemini/Qwen natijalari alohida **gipotetik** taqqoslashdir: har
bir yo'nalish uchun model javobi tayyor bo'lgandagi MT5 narxi, saqlangan ATR
va 24 soatlik muddat ishlatiladi. Modellarning boshlanish vaqti va narxi
farq qilishi mumkin; bu aynan bir vaqtda kirilgan juft benchmark emas.
Qwen faqat yo'nalish beradi, shuning uchun unda javobi tayyor bo'lgach kirish
faraz qilinadi; Gemini kutishni aytganda uning rejasi hisoblanmaydi. Ochiq,
ma'lumotsiz va baholanmagan holatlar TP ulushiga kirmaydi. Modellar namunalari
soni farq qilishi mumkin. Eski tahlillar qayta tahlil qilinmaydi va ular uchun
yangi gipotetik rejalar to'qib yaratilmaydi.

Baholash shartlari reja bilan saqlanadi: 24 soat, UTC+0. `/evaluate` so'rovi
ularni o'zgartira olmaydi; boshqa parametrlar 400 xato qaytaradi.
Shartlari saqlanmagan eski yakuniy natijalar legacy hisoblanadi.
Baholash 24 soat davomida Bid/Ask tick tarixini tekshiradi. Ko'rinadigan
60 soniyadan katta uzilishda natija ma'lumotsiz qoladi; broker har bir tickni
yetkazganini to'liq isbotlab bo'lmaydi. Komissiya va slippage hisoblanmaydi.
Eski usulda baholangan natijalar tarixda saqlanadi, yangi statistikaga qo'shilmaydi.
Model bahosi haqiqiy savdo muvaffaqiyati ehtimoli sifatida tasdiqlanmagan.
Modellar bir xil MT5 ma'lumotlaridan ham foydalanadi; ularning kelishuvi
mustaqil tasdiq yoki foyda kafolati emas.

Tahlil oxirida narx yangiligi qayta tekshiriladi. Shamlar keshi ham joriy
sham vaqti orqali tekshiriladi; odatiy dam olish/tatil uchun 7 kungacha
uzilish ruxsat etiladi, uzoqroq uzilishda kontekst qabul qilinmaydi.
Kotirovka 60 soniyadan eski,
narx tahlil davomida 0.25 ATR dan ko'p o'zgargan yoki Gemini kirishni kutishni
aytgan bo'lsa Hold qaytariladi. Kutish sharti bajarilganda qayta tahlil qiling.
Eski fayl vaqti aniqlansa Hold qaytariladi va tahlil signal statistikasiga
qo'shilmaydi. Fayl vaqti rasm olingan vaqtni kafolatlamaydi: chart joriyligini
tekshiring. Yuklash chegarasi 10 MiB va 25 million piksel.
API faqat lokal manzilga ochiq; barcha POST so'rovlarida
`X-Grid-Bot-Request: 1` sarlavhasi talab qilinadi.
Windows kirishida avtomatik ishga tushirish faqat foydalanuvchi tizimga kirganda
ishlaydi; MT5 ochiq va hisobga ulangan bo'lishi alohida talab qilinadi.

## Modelni xavfsiz tadqiq qilish

Yangi train/validation/test split'larida barcha symbol va timeframe'lar uchun
umumiy vaqt chegaralari ishlatiladi; label davri chegarani kesib o'tgan qatorlar
chiqariladi. Eski multi datasetni mavjud raw fayllardan yangilang, keyin candidate
modelni o'qiting:

```powershell
python data_pipeline/relabel_from_raw.py --variant all
python training/train_numeric_model.py
python training/backtest.py
```

Birinchi buyruq `multi` va `multi_v2` label fayllarini avval `backups/` ichiga
zaxiralab, keyin qayta yozadi. Training faol backend modelini almashtirmaydi: natija
`training/model_candidate/` papkasiga yoziladi. Backtest dastlab faqat
validation bo'yicha parametrlar jadvalini chiqaradi. Shu yerda tanlangan
qiymatlarni mustaqil test uchun aniq kiriting, masalan:

```powershell
python training/backtest.py --split test --confidence 0.55 --tp-ratio 2
```

Bu qiymatlar misol, tavsiya yoki tasdiqlangan strategiya emas. Test bo'yicha
qayta-qayta parametr tanlash mustaqil testni buzadi. Backtest dataset va candidate
model hash'larini tekshiradi, faqat custom modelni simulyatsiya qiladi; Gemini,
spread, komissiya va slippage hisoblanmaydi. Muddat tugagan pozitsiyalar faraz
qilingan so'nggi yopilish narxida alohida ko'rsatiladi.

Faol eski model metadata'sida `legacy_unverified` belgisi bor: fayl hash'i
aniq, lekin uning dastlabki training manbalarini qayta isbotlab bo'lmaydi.
Yangilangan tarixiy split ham avval ko'rilgan davrlardan tuzilgan; yangi,
mustaqil holdout hisoblanmaydi. Candidate avtomatik deploy qilinmaydi.

Regressiya testlari (haqiqiy broker va pullik API chaqirilmaydi):

```powershell
python -m pytest backend training/test_research.py -q
node --test frontend/test_app.cjs
```
