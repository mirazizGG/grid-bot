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

## Yozuvlar

_(hali yozuv yo'q -- birinchi natijani kutyapmiz)_

## Haftalik xulosa (1 hafta to'lgach shu yerga yoziladi)

_(hali to'ldirilmagan)_
