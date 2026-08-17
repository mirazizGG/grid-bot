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

## Haftalik xulosa (1 hafta to'lgach shu yerga yoziladi)

_(hali to'ldirilmagan)_
