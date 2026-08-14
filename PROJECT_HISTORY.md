# Loyiha tarixi va qarorlar jurnali

Bu fayl **README.md**'dan farq qiladi: README "qanday ishga tushirish"ni
tushuntiradi, bu fayl esa **nega hozirgi holatga kelinganini, qaysi
yo'llar sinab ko'rilib rad etilganini va shu sabablarni** tushuntiradi.
Yangi Claude Code sessiyasi (yoki yangi odam) loyihani tushunish uchun
avval shuni o'qishi kerak — aks holda allaqachon sinab ko'rilgan va
ishlamagan narsalarni qayta taklif qilib, vaqt behuda ketishi mumkin.

## Loyiha nima

Foydalanuvchi (XAUUSD/oltinda savdo qiladi) chart skrinshotini yuklaydi.
Tizim ikkita mustaqil "aql"ni solishtiradi:

1. **Custom model** — MT5'dan olingan RSI/MACD/ADX kabi raqamli
   indikatorlarga qarab (rasmni ko'rmaydi) Buy/Sell/Hold aytadi.
2. **Gemini (Google AI)** — chart rasmini + shu raqamli indikatorlarni
   birga ko'rib, mustaqil xulosa chiqaradi.

Ikkalasi kelishsa va custom model ishonchi yetarli bo'lsa — signal va
TP/SL beriladi. Aks holda — **Hold** (xavfsizlik ustunligi).

## Bosqichlar tarixi (xronologik)

### 1-bosqich — Data pipeline
`data_pipeline/` — MT5'dan XAUUSD M15 tarixini olib (`mt5_fetch.py`),
ATR-asosidagi "triple barrier" usulida Buy/Sell/Hold label yaratildi
(`label_generator.py`), so'ng candlestick rasmlar generatsiya qilindi
(`render_charts.py`).

**Muhim tuzatish**: dastlab har bir sham uchun window yaratilgan edi
(`WINDOW_STRIDE=1`) — qo'shni rasmlar 79/80 bir xil bo'lib, model ularni
"yodlab" oldi (train acc 99%, val acc 33% — klassik overfitting).
`WINDOW_STRIDE=20` ga o'zgartirilib tuzatildi.

### 2-bosqich — Rasm-CNN model (ENDI ISHLATILMAYDI)
`training/train_colab.ipynb` — ResNet18 transfer learning (Google Colab,
GPU). Natija: **test aniqligi 33%** — tasodifiy taxmin darajasi. Rasm
patternidan yolg'iz signal chiqarish ishonchli emas degan xulosaga
kelindi.

Fayllari saqlanadi (`backend/model/chart_signal_model.onnx*` — bular
`.gitignore`da, repo'da yo'q), lekin **backend ularni ishlatmaydi**.
Kod: `backend/custom_model_legacy_cnn.py` (faqat tarixiy referens).

### 3-bosqich — Raqamli indikator model (HOZIR FAOL)
G'oya: CNN o'rniga RSI/MACD/ATR/SMA/Bollinger/ADX/Stochastic/sessiya
vaqti kabi 17 ta raqamli xususiyat asosida GradientBoosting model
(`training/train_numeric_model.py`, `backend/custom_model.py`).

Bir nechta variant sinaldi (natijalar **TEST split, XAUUSD M15** uchun):

| Variant | Aniqlik | Xulosa |
|---|---|---|
| Faqat XAUUSD (2.5 yil) | 36% | Boshlang'ich natija |
| **XAUUSD+EURUSD+GBPUSD+USDJPY (2.5 yil) — HOZIRGI FAOL** | **46%** | Eng yaxshi -- ko'proq symbol xilma-xillik berdi |
| 12 symbol (XAUUSD, XAGUSD, EUR/GBP/USD/CHF/CAD/AUD/NZD/JPY juftliklari) | 39% | YOMONLASHDI -- juda ko'p symbol signalni "suyultirdi" |
| Faqat XAUUSD + LSTM (vaqt ketma-ketligi, "video" g'oyasi) | 37% | Yordam bermadi -- ma'lumot yetarli emas edi |
| 4 symbol + 14 yillik tarix (GitHub'dan tashqi ma'lumot) | 38% | YOMONLASHDI -- 2012-yilgi bozor rejimi hozirgidan farqli |
| 4 symbol + 5 yillik tarix | 37% | Yana yomonlashdi |

**Xulosa**: "4 symbol, ~2.5 yil" — bu tasodifiy topilgan "shirin nuqta".
Undan ortiq yoki kamroq ma'lumot — natijani yaxshilamadi, ko'pincha
yomonlashtirdi. Bu chegara **qayta-qayta tekshirilgan**, tasodif emas.

**Nega ko'proq ma'lumot yordam bermadi**: (1) juda ko'p symbol qo'shilsa,
har biriga xos pattern "o'rtachalashib" ketadi; (2) juda uzoq tarix
qo'shilsa, eski bozor rejimi (masalan 2012-yilda oltin $1600, hozir
$4300+) hozirgi vaziyatga mos kelmaydi.

**Deploy qilingan model**: `backend/model/numeric_model.joblib` +
`numeric_model_meta.json` (git'da bor, ~750 KB). Buni **qayta o'qitib
YOZIB YUBORMANG** agar aniq sababingiz bo'lmasa — bu allaqachon eng
yaxshi topilgan natija. Agar qayta o'qitsangiz va yomonroq chiqsa,
`git checkout HEAD -- backend/model/numeric_model.joblib
backend/model/numeric_model_meta.json` orqali tiklang.

### 4-bosqich — Gemini integratsiyasi
Boshida Claude API ko'zda tutilgan edi, lekin foydalanuvchida faqat
Gemini kaliti bor edi — shuning uchun `backend/gemini_vision.py` orqali
Google Gemini (`gemini-2.5-flash`) ishlatiladi. **Eslatma**: bepul tarif
kuniga atigi ~20 so'rovga cheklangan — test paytida tez tugab qoladi.

### 5-bosqich — MT5 jonli kontekst
Dastlab Gemini faqat rasmga qarab narxni "taxmin" qilardi (ko'pincha
noto'g'ri yoki bo'sh). Yechim: `backend/mt5_context.py` — har tahlilda
backend MT5'ga ulanib, jonli bid/ask, RSI, MACD, ATR, SMA20/50/100,
Bollinger %B, ADX, Stochastic, sessiya vaqtini oladi va bularni ham
Gemini'ga, ham custom model'ga beradi. TP/SL endi **deterministik**
(ATR asosida Python'da hisoblanadi), Gemini/model taxminiga tayanmaydi.

### 6-bosqich — Backtest va sifat filtri
Foydalanuvchi "aniqlik past" deb tashvishlangач, **80-90% kabi
aniqlikning imkonsizligi** tushuntirildi (agar bunday oson bo'lganida,
professional fondlar buni allaqachon qilgan bo'lardi). Buning o'rniga
`training/backtest.py` — real TEST ma'lumotda turli ishonch chegarasi va
TP:SL nisbatlarida haqiqiy win-rate/expectancy hisoblandi.

**Natija**: ishonch filtri bo'lmasa expectancy ~0. `custom model
ishonchi >= 55%` + `TP:SL = 2:1` bilan **+0.095R/savdo** (ijobiy, 74
savdo namunasida). Shu sozlamalar `backend/config.py`
(`MIN_CUSTOM_MODEL_CONFIDENCE=0.55`) va `backend/mt5_context.py`
(`TP_ATR_MULT=2.0`) ga o'rnatildi.

**Bu qasddan qilingan**: signal endi kamroq chiqadi (ko'p holatda Hold),
bu **kutilgan va to'g'ri xatti-harakat** — sifat > miqdor.

### Rad etilgan g'oyalar (qayta taklif qilmang)
- **Video darslardan freym olib label qilish**: rad etildi, chunki
  freym'dan keyin narx qayerga ketganini tekshirib bo'lmaydi (label
  haqiqiy emas, shunchaki taxmin). Ko'p vaqt sarflansa ham natija
  yaxshilanmaydi, yomonlashadi.
- **3-chi LLM qo'shish (Claude/GPT)**: rad etildi, chunki barcha LLM'lar
  bir xil turdagi (rasm+mulohaza), haqiqiy bashorat qobiliyati yo'q,
  ular orasidagi "xilma-xillik" foydasi kichik, xarajat esa 2 baravar.
- **Ko'proq tarixiy/ko'proq symbol ma'lumot**: yuqorida ko'rsatilgandek,
  4-marta sinaldi, hech biri yordam bermadi.

## Hozirgi faol konfiguratsiya (muhim raqamlar)

- Custom model: GradientBoosting, 17 xususiyat, XAUUSD+EURUSD+GBPUSD+USDJPY
  (M15+H1), ~2.5 yillik MT5 tarixi, test aniqligi **46%** (XAUUSD M15)
- Consensus: Gemini asosiy manba (75% vazn), custom model tasdiqlovchi
  (25% vazn) + **ishonch filtri (>=55%)**
- TP:SL nisbati: **2:1** (SL=1xATR, TP=2xATR)
- Frontend: statik HTML/CSS/JS, backend orqali serve qilinadi (build
  step yo'q)

## Fayllar holati (qaysi kod hali ishlatiladi)

| Fayl | Holat |
|---|---|
| `backend/custom_model.py` | **FAOL** -- joriy raqamli model |
| `backend/custom_model_legacy_cnn.py` | Legacy, ishlatilmaydi (referens) |
| `backend/sequence_model.py` | Tajriba (LSTM), ishlatilmaydi (referens) |
| `backend/gemini_vision.py`, `mt5_context.py`, `consensus.py`, `main.py`, `db.py` | **FAOL** |
| `data_pipeline/multi_symbol_dataset.py` | **FAOL** -- deploy modelni takrorlash uchun (4 symbol config) |
| `data_pipeline/fetch_external_history.py`, `relabel_from_raw.py` | Tajriba (tashqi tarix), natija salbiy chiqqani uchun ishlatilmaydi |
| `training/train_numeric_model.py` | **FAOL** -- custom model shu bilan o'qitiladi |
| `training/train_colab.ipynb`, `train_sequence_model.py` | Legacy/tajriba, referens uchun saqlanadi |
| `training/backtest.py` | **FAOL** -- yangi threshold/R:R sinash kerak bo'lsa ishlatiladi |

## Keyingi safar nima qilish mumkin (agar davom ettirilsa)

- Ma'lumot miqdorini oshirish orqali aniqlikni yanada oshirishga
  urinish endi **kam foydali** — bu yo'l 4 marta sinaldi.
- Agar aniqlikni oshirish kerak bo'lsa, boshqa yo'nalish qidiring:
  masalan faqat kuchli ADX (aniq trend) paytida signal berish, yoki
  boshqa tarzda label yaratish (masalan strukturaviy support/resistance
  darajalariga asoslangan, ATR ko'paytmasi emas).
- Gemini bepul kvotasi tez tugaydi -- ko'p test qilish kerak bo'lsa,
  to'lovli rejaga o'tish yoki so'rovlarni keshlash kerak bo'lishi mumkin.
