# 1-haftalik sinov jurnali (consensus-tuzatish keyingi kuzatuv)

Bu fayl **2026-08-15**da boshlangan, 1 haftalik amaliy sinov davri uchun.
Maqsad: `backend/consensus.py`dagi so'nggi tuzatish (custom model va Gemini
qarama-qarshi bo'lganda, Gemini ishonchi >=65% bo'lsa Gemini signaliga
tayanish) real foydalanishda qanday natija berayotganini kuzatish.

Foydalanuvchi har safar tahlil natijasini (screenshot yoki tavsif)
yuborganda, shu yerga yozib boriladi -- shu bilan boshqa kompyuterdan kirilsa
ham (yoki Claude Code sessiyasi qayta boshlansa ham) butun tarix va xulosalar
yo'qolmaydi.

## Kuzatilayotgan narsa

- Final signal to'g'ri chiqyaptimi (keyinchalik narx qayerga ketganiga qarab)
- `agreement=False` holatlar (custom model va Gemini kelishmagan) qanday
  yakunlanmoqda -- ayniqsa Gemini ustunlik qilib signal bergan holatlar
- Custom modelning yangi matnli izohi (`reasoning.py`) foydali/tushunarli
  ko'rinyaptimi

## Format (har bir yozuv uchun)

```
### [sana] -- [symbol] [timeframe]
- Final signal: ...
- Custom model: ... (ishonch: ...%)
- Gemini: ... (ishonch: ...%)
- Kelishuv: ha/yo'q (agar yo'q -- Gemini ustunlik qildimi?)
- Keyinchalik natija (agar ma'lum bo'lsa): to'g'ri chiqdi / xato chiqdi / hali noma'lum
- Claude xulosasi: ...
```

## ⚠️ Muhim metodologik eslatma (2026-08-17)

Foydalanuvchi **eski (oldin olingan) chart screenshot**larini yuklab sinov
qilishga urindi va natijani screenshot olingan davrdan keyin narx qayerga
ketgani bilan solishtirmoqchi bo'ldi. **Bu noto'g'ri test usuli** -- sababi:

- Custom model rasmni umuman ko'rmaydi -- faqat MT5'dan **hozirgi jonli**
  RSI/MACD/ADX/Entry narxini oladi. Eski rasm undan mutlaqo mustaqil.
- Gemini rasmni ko'radi, lekin unga ham **hozirgi** (rasm vaqtiga mos
  kelmaydigan) raqamli kontekst beriladi -- bu uni chalg'itishi mumkin.
- Natijada "Entry: 4399.84" kabi narxlar rasm olingan paytdagi emas,
  so'rov yuborilgan paytdagi narx bo'ladi.

**Xulosa**: eski rasm + "keyin narx qayerga ketdi" solishtiruvi tizim
sifatini baholash uchun **yaroqsiz** -- ikki xil vaqt davri solishtirilyapti.
To'g'ri test uchun: MT5'da **hozir** ochilgan screenshot darhol yuklanishi,
so'ng natija bilan **keyingi 1-4 soatlik** haqiqiy narx harakati solishtirilishi
kerak. Shu qoidaga rioya qilinmagan yozuvlar quyida "(eski rasm -- yaroqsiz test)"
deb belgilanadi va haftalik xulosaga kiritilmaydi.

## Yozuvlar

### 2026-08-17 -- XAUUSD M15 (eski rasm -- yaroqsiz test)
- Final signal: Hold (ishonch 50%)
- Custom model: Sell (ishonch: 46%)
- Gemini: Sell (ishonch: 60%)
- Kelishuv: ha yo'nalishda (ikkalasi ham Sell), lekin custom ishonchi past
  (46% < 55%) bo'lgani uchun tizim Hold'ga o'tdi
- Keyinchalik natija: solishtirib bo'lmaydi -- foydalanuvchi eski (oldin
  olingan) screenshot yukladi, natija esa hozirgi jonli narxga asoslangan.
  Yuqoridagi metodologik eslatmaga qarang.
- Claude xulosasi: Bu holat o'zi natija sifatini baholash uchun ishlatib
  bo'lmaydi, lekin consensus mantig'i to'g'ri ishlagani ko'rinadi: ikkala
  model bir yo'nalishda (Sell) kelishdi, faqat custom model ishonchi
  chegaradan (55%) past bo'lgani uchun xavfsizlik tufayli Hold berildi --
  bu kutilgan/loyihalangan xulq-atvor, xato emas. Foydalanuvchidan **hozirgi
  vaqtda** olingan screenshot bilan qayta test so'raldi.

### 2026-08-17 -- XAUUSD M15
- Final signal: Hold (ishonch 50%)
- Custom model: Buy (ishonch: 49%)
- Gemini: Buy (ishonch: 70%)
- Kelishuv: ha yo'nalishda (ikkalasi ham Buy), lekin custom ishonchi past
  (49% < 55%) bo'lgani uchun tizim Hold'ga o'tdi
- Keyinchalik natija: hali noma'lum (kuzatib boriladi)
- Claude xulosasi: Yana o'sha 55% chegara holati -- ikkala model bir xil
  yo'nalishda (Buy), Gemini ishonchi baland (70%), custom model chegaraga
  juda yaqin (49%, atigi 6 punkt kam). Consensus mantiqan to'g'ri ishladi,
  lekin bu ikkinchi marta ketma-ket shu xil holat ("ikkalasi rozi, custom
  ishonchi chegaradan sal past -> Hold") -- agar bu naqsh davom etsa, 55%
  chegara biroz qattiqroq bo'lishi mumkinligini (masalan 50%ga tushirish)
  hafta oxirida ko'rib chiqish kerak.
- **Yangilanish (2026-08-18, MT5 tarixi bilan tekshirildi)**: Entry
  (4401.53)dan keyin narx bir necha bar ichida 4377gacha tushib, **ikkala
  model SL darajasini ham (custom 4395.70, Gemini 4394.00) buzgan** --
  keyinroq narx kuchli ko'tarilib 4436gacha chiqqan bo'lsa-da, bu allaqachon
  SL urilgandan keyingi harakat, ahamiyatsiz. **Agar Buy savdo qilingan
  bo'lsa -- zararli bo'lardi.** Tizimning yakuniy Hold qarori (custom model
  ishonchi 55% chegaradan past bo'lgani sabab) bu safar foydali bo'ldi --
  ikkala model yo'nalishda kelishgan bo'lsa ham, yo'nalish o'zi noto'g'ri
  chiqdi. Claude xulosasi: bu 55% ishonch chegarasini pasaytirish g'oyasiga
  qarshi dalil -- aksincha, past ishonchli custom model signalini e'tiborsiz
  qoldirish (hozirgi xulq) to'g'ri strategiya ekanini tasdiqladi.

### 2026-08-18 -- XAUUSD M15 (qarama-qarshi holat)
- Final signal: Hold (ishonch 50%)
- Custom model: Sell (ishonch: 42%)
- Gemini: Buy (ishonch: 50%)
- Kelishuv: yo'q (Buy vs Sell ziddiyat), Gemini ishonchi ham 65% override
  chegarasidan past (50% < 65%) -- shuning uchun xavfsizlik uchun Hold
- Keyinchalik natija: hali noma'lum (kuzatib boriladi)
- Claude xulosasi: Bu -- 7-bosqichda qo'shilgan Gemini-override mantig'ining
  "ishlamagan" tarafi: Gemini ishonchi 65% chegaraga yetmagani uchun override
  qilmadi, to'g'ri xavfsizlik choralari ishladi. E'tiborli narsa: ikkala model
  ham o'zaro zaif ishonch bilan (42%, 50%) qarama-qarshi signal berdi --
  RSI/Stochastic bo'yicha oversold ko'rinishi bor (custom: RSI 30, Stoch 13),
  bu odatda "pastga trend davom etadi" (Sell) va "tez orada tiklanish keladi"
  (Buy) degan ikki xil talqinni tug'diradigan klassik noaniq holat. Hold
  qarori mantiqan asosli.

### 2026-08-18 -- XAUUSD M15 (#2)
- Final signal: Hold (ishonch 50%)
- Custom model: Buy (ishonch: 37%)
- Gemini: Buy (ishonch: 70%)
- Kelishuv: ha yo'nalishda (ikkalasi Buy), lekin custom ishonchi past
  (37% < 55%) -- Hold
- Keyinchalik natija: hali noma'lum (kuzatib boriladi)
- Claude xulosasi: Bu -- 3-marta ketma-ket "ikkalasi rozi (Buy), custom
  ishonchi 55% chegaradan past, Gemini 70% yaqin" naqshi (17-avgust va
  shu kungi #1 bilan solishtiring). Farqi: bu safar custom ishonchi ancha
  pastroq (37%, oldingilarida 46-49% edi) -- ya'ni custom model o'zi ham
  o'z signalidan unchalik ishonchli emas. 17-avgustdagi shunga o'xshash
  holat (Buy/Buy, past custom ishonch) keyinchalik SL urilib yomon chiqqan
  edi -- agar bu naqsh yana takrorlansa, "ikkalasi Buy/Sell'da rozi, lekin
  custom ishonchi past" holatlarining haqiqiy statistikasi hafta oxirida
  alohida ko'rib chiqilishi kerak (hozircha 55% chegara mantiqan to'g'ri
  ishlab turibdi -- oldingi holat buni tasdiqladi).

## Haftalik xulosa (1 hafta to'lgach shu yerga yoziladi)

_(hali to'ldirilmagan)_
