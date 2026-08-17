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
| 4 symbol + struktura/sham-pattern feature qo'shildi (trend_slope, swing HH/HL, candle body/wick — 17→22 feature) | 35.3% | YOMONLASHDI -- qo'shimcha feature'lar kam ma'lumotda shovqin qo'shdi, bekor qilindi (git checkout) |

**Xulosa**: "4 symbol, ~2.5 yil, 17 feature" — bu tasodifiy topilgan "shirin nuqta".
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

### 7-bosqich — Gemini ustunlik qilishi (qarama-qarshi holatda)
Foydalanuvchi real holatda ko'rdi: custom model Sell (47% ishonch),
Gemini Buy (70% ishonch) dedi — narx aslida yuqoriga ketdi (Gemini to'g'ri
chiqdi). Avvalgi mantiqda bunday **to'g'ridan-to'g'ri ziddiyat (Buy vs
Sell)** har doim Hold'ga olib kelardi — bu holatda to'g'ri Gemini signali
ham yo'qolib ketardi.

**Sabab tushuntirildi**: custom model (46% aniqlik, deyarli tasodifiy)
Gemini bilan solishtirib bo'lmaydigan darajada oddiy tizim — Gemini
milliardlab parametrli, umumiy "tushunish"ga ega LLM, custom model esa
faqat 17 ta raqamli ko'rsatkichga asoslangan kichik GBM. Custom modelni
"Gemini darajasiga" olib chiqish talab qiladigan resurs (ulkan ma'lumot +
GPU + oylab tajriba) hozirgi loyiha doirasida imkonsiz — foydalanuvchiga
tushuntirilib, rad etildi (pastdagi ro'yxatga qarang).

**Yechim**: `backend/consensus.py` — signal to'g'ridan-to'g'ri
qarama-qarshi (Buy vs Sell) bo'lganda, agar Gemini ishonchi yetarlicha
yuqori bo'lsa (`MIN_VISION_OVERRIDE_CONFIDENCE=0.65`,
`backend/config.py`), custom model ovozi e'tiborga olinmaydi va Gemini
signaliga tayaniladi (`agreement=False`, lekin signal beriladi, izohda
ogohlantirish bilan). Gemini ishonchi past bo'lsa, hamon Hold (ikkalasi
ham noaniq).

### 8-bosqich — Custom modelga qoidaga asoslangan izoh (reasoning)
Foydalanuvchi custom modelning ham Gemini kabi "fikrlab" matnli javob
berishini so'radi. Custom modelni haqiqiy LLM qilib bo'lmasligi
tushuntirilgandan so'ng, oraliq yechim tanlandi: `backend/reasoning.py` --
custom model chiqargan signal (Buy/Sell/Hold) qaysi indikator qiymatlari
(RSI, MACD, ADX, Stochastic, SMA trend) bilan bog'liqligini qoidaga
asoslangan (rule-based, LLM emas) matnga aylantiradi va Gemini'nikiga
o'xshash `trend`/`reasoning` maydonlarini qaytaradi.

**Muhim**: bu **aniqlikni oshirmaydi** (custom model hamon ~46%) -- faqat
signalni **tushunarli qiladi**. `backend/custom_model.py` endi
`reasoning.explain()`ni chaqiradi, frontend (`frontend/app.js`) custom model
kartasida ham shu matnni ko'rsatadi.

### 9-bosqich — 1-haftalik amaliy kuzatuv (2026-08-15 dan)
Foydalanuvchi 3-4 chi LLM (GPT/Claude/Grok) qo'shishni taklif qildi --
tushuntirildi: bularning aksariyati bepul API tarifiga ega emas, va
qo'shimcha LLM narxni bashorat qilish tub muammosini hal qilmaydi (barcha
LLM'lar bir xil turdagi, xilma-xillik kam). Kelishildi: 1 hafta davomida
pul sarflamasdan, hozirgi tuzatilgan consensus (7-8-bosqich) qanday
ishlayotganini kuzatamiz. Har bir natija **`TESTING_LOG.md`**ga yoziladi
(foydalanuvchi natija yuboradi -> Claude shu faylga yozadi va xulosa
qo'shadi) -- shu bilan boshqa kompyuterdan kirilsa ham davomiylik
saqlanadi. 1 hafta so'ng shu jurnal asosida keyingi qadam (masalan
pullik ikkinchi LLM qo'shish yoki hozirgidek qolish) hal qilinadi.

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
- **Custom modelni "Gemini kabi fikrlaydigan" qilish** (struktura/sham-pattern
  feature qo'shish orqali): sinaldi, 46%dan 35.3%ga tushirdi, bekor qilindi.
  Custom model va Gemini fundamental jihatdan boshqa toifadagi tizimlar --
  biri kichik statistik model (~8000 qator ma'lumot), ikkinchisi milliardlab
  parametrli LLM. Custom modelni Gemini darajasiga olib chiqish uchun kerak
  bo'ladigan resurs (ma'lumot+compute) loyiha doirasida yo'q -- bu
  tushuntirildi. Buning o'rniga consensus mantig'ida Gemini ustunlik qiladi
  (7-bosqichga qarang).

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

### 10-bosqich — Algoritm/hyperparameter tajribalari (2026-08-17)

Foydalanuvchi so'rovi bilan custom model aniqligini oshirish uchun bir nechta
algoritmik yo'l sinaldi (`training/tune_hyperparams.py`, `tune_v2.py`,
`tune_v3_binary.py` -- barchasi repo'da qoldirilgan, qayta ishlatish mumkin):

1. **Optuna hyperparameter qidiruv** (150 trial, umumiy VAL accuracy'ga
   optimallashtirilgan) -- Hold klassini butunlay yo'q qildi, XAUUSD M15
   TEST'ni 46%dan 34.7%ga tushirdi. Bekor qilindi.
2. **4 xil model oilasi** (sklearn GBM, HistGradientBoosting, LightGBM,
   XGBoost) — har biri balanslangan va balanslanmagan `class_weight` bilan
   sinaldi. **Balanslash barcha 4 oilada ham izchil yomonlashtirdi**
   (29-35% oralig'ida) — bu yagona izchil, ishonchli xulosa.
   Balanslanmagan versiyalar orasida aniq g'olib chiqmadi (barchasi
   38-40% atrofida, joriy modeldan yaxshi emas).
3. **2-klass qayta formatlash** (Hold'ni o'qitishdan chiqarib, faqat
   Buy/Sell farqini o'rgatib, ishonch pastligida Hold qo'yish) — turli
   threshold sinaldi, hech biri joriy modeldan yaxshi chiqmadi (threshold
   oshgani sari Hold haddan tashqari ko'p bashorat qilinib, aniqlik
   pasaydi: 40%dan 21%gacha).

**Muhim kashfiyot**: shu tajribalar davomida, xuddi shu kod + xuddi shu
parametr + xuddi shu random_state bilan modelni qayta o'qitish **hujjatlashtirilgan
46%ni qayta bermadi** (39% chiqdi). Sabab topildi: `backend/requirements.txt`da
`scikit-learn>=1.5` va `numpy>=1.26` qattiq versiyaga bog'lanmagan edi;
`pip install optuna/lightgbm/xgboost` fonda ularni yangilab yubordi
(scikit-learn 1.9.0, numpy 2.4.4), bu esa GradientBoostingClassifier'ning
random_state bilan bog'liq ichki xatti-harakatini o'zgartirdi. **Tuzatildi**:
`requirements.txt`da endi `scikit-learn==1.9.0` va `numpy==2.4.4` qattiq
belgilangan -- kelajakda shunday drift qayta bo'lmasligi uchun.

Bundan tashqari shuni ko'rsatdiki: XAUUSD M15 TEST subset atigi **150 qator**
-- bunday kichik namunada bir necha foizlik farq (masalan 38% vs 46%) katta
ehtimol bilan shunchaki tasodifiy shovqin, haqiqiy yaxshilanish/yomonlashish
emas. Kelajakda solishtirish qilinganda to'liq TEST (1200 qator) yoki
bir nechta random_state bo'yicha o'rtacha olish tavsiya etiladi.

**Xulosa**: hech bir algoritmik/hyperparameter o'zgarishi joriy deploy
qilingan modeldan ishonchli ravishda yaxshi chiqmadi. Deploy qilingan model
**o'zgartirilmadi** (git holatida qoldi). Custom model performance plafoni
(~40-46% atrofi, aniq raqam kichik test hajmi tufayli noaniq) hozirgi 17
feature + ~8000 qatorlik data bilan algoritm darajasida yechilmaydigan
chegara ko'rinadi -- yagona qolgan real yo'nalish **ko'proq/boshqacha data**
(masalan uzoqroq tarix yoki boshqa label metodologiyasi), algoritm emas.

### 11-bosqich — 5x ko'proq tarixiy data bilan alohida model (2026-08-17)

Foydalanuvchi talabi: joriy deploy qilingan modelga tegmasdan, ancha kattaroq
datasetdan **alohida yangi model** yaratib solishtirish.

- `data_pipeline/multi_symbol_dataset_v2.py`: MT5'dan har symbol/timeframe
  uchun oldingi 20,000 bar o'rniga **99,999 bar** (terminal `maxbars=100000`
  chegarasi) tortib olindi -- XAUUSD H1/EURUSD H1/GBPUSD H1/USDJPY H1 uchun
  ~2010-yildan buyon (~16 yil), M15 juftliklar uchun broker tarixi cheklovi
  tufayli ~2022-yildan buyon (~4 yil). Natija: **39,960 qator** (avvalgi
  7,960dan 5x ko'p).
- `training/train_numeric_model_v2.py`: xuddi shu feature pipeline
  (`mt5_context.py`, train/serve parity buzilmadi), xuddi shu
  GradientBoostingClassifier parametrlari bilan o'qitildi, natija
  `backend/model/numeric_model_v2.joblib` ga saqlandi (**deploy qilingan
  `numeric_model.joblib`ga tegilmadi**).

**Natija**: XAUUSD M15 TEST = **36.9%** (n=750), umumiy TEST = 39.1% (n=6000).
Joriy (kichik data, ~8k qator) modelning shu kunda xolis o'lchangan bazaviy
natijasi (39.3% XAUUSD M15, n=150) bilan solishtirganda -- **sezilarli farq
yo'q**, agar biror narsa bo'lsa ozgina yomonroq. Muhimi, bu safar test hajmi
ancha katta (750 vs 150), shuning uchun bu natija avvalgilariga qaraganda
ancha ishonchliroq: **5x ko'proq tarixiy data aniqlikni oshirmadi**.

**Umumiy xulosa (10 va 11-bosqichlar birgalikda)**: 6 ta mustaqil yo'nalish
sinaldi -- hyperparameter tuning, 4 xil algoritm oilasi, class balancing,
2-klass reformatlash, va endi 5x ko'proq data -- **birortasi ham ishonchli
ravishda yaxshilanish bermadi**. Bu custom modelning ~35-40% aniqlik
darajasi tasodifiy emas, balki **hozirgi 17 ta indikator-feature + ATR
triple-barrier label dizayni bilan real chegara** ekanini ko'rsatadi.
Buni yengish uchun endi faqat **butunlay boshqacha yondashuv** kerak bo'lishi
mumkin (masalan butunlay boshqa label metodologiyasi, yoki narx bashorati
o'rniga boshqa maqsad funksiyasi) -- shu chegara doirasida yana ma'lumot yoki
algoritm bilan "kuchaytirish" endi oqilona emas.

Fayllar: `data_pipeline/output/multi_v2*`, `backend/model/numeric_model_v2.*`
repo'da **saqlanmaydi** (`.gitignore`, katta va qayta generatsiya qilinadigan) --
faqat skriptlar (`multi_symbol_dataset_v2.py`, `train_numeric_model_v2.py`)
versiyalanadi, kerak bo'lsa qayta ishga tushirish mumkin.

## Keyingi safar nima qilish mumkin (agar davom ettirilsa)

- Ma'lumot miqdorini oshirish orqali aniqlikni yanada oshirishga
  urinish endi **kam foydali** — bu yo'l 4 marta sinaldi.
- Agar aniqlikni oshirish kerak bo'lsa, boshqa yo'nalish qidiring:
  masalan faqat kuchli ADX (aniq trend) paytida signal berish, yoki
  boshqa tarzda label yaratish (masalan strukturaviy support/resistance
  darajalariga asoslangan, ATR ko'paytmasi emas).
- Gemini bepul kvotasi tez tugaydi -- ko'p test qilish kerak bo'lsa,
  to'lovli rejaga o'tish yoki so'rovlarni keshlash kerak bo'lishi mumkin.
