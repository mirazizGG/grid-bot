# Bosqich 2 — Model o'qitish (Google Colab)

## 1. Dataset'ni Google Drive'ga yuklash

Loyiha ildizida `dataset_grid_bot.zip` (~300+ MB) fayli tayyorlangan
(`dataset/images/` + `manifest.csv` ichida).

1. https://drive.google.com ga kiring.
2. `MyDrive` ichida `grid-bot` papkasini yarating.
3. `dataset_grid_bot.zip` faylini shu papkaga yuklang (`MyDrive/grid-bot/dataset_grid_bot.zip`).

## 2. Notebook'ni Colab'da ochish

1. https://colab.research.google.com ga kiring.
2. `File > Upload notebook` orqali `training/train_colab.ipynb` faylini yuklang.
3. `Runtime > Change runtime type > Hardware accelerator > GPU (T4)` ni tanlang.
4. Cell'larni tartib bilan yuqoridan pastga ishga tushiring:
   - Google Drive ulanadi (ruxsat so'raladi — tasdiqlang).
   - `dataset_grid_bot.zip` `/content/dataset` ga ochiladi.
   - Model (ResNet18, ikki boshli: Buy/Sell/Hold klassifikatsiya + TP/SL regressiya) o'qitiladi (~15 epoch, T4'da taxminan 20-40 daqiqa).
   - Test to'plamida aniqlik (accuracy), precision/recall va confusion matrix chiqariladi.
   - `chart_signal_model.onnx` va `norm_stats.json` `MyDrive/grid-bot/model_output/` ga saqlanadi.

## 3. Natijani lokal loyihaga qaytarish

O'qitish tugagach, quyidagi ikkita faylni Google Drive'dan yuklab oling va
loyihaga qo'ying (Bosqich 3 backend shularni ishlatadi):

```
MyDrive/grid-bot/model_output/chart_signal_model.onnx  -> backend/model/chart_signal_model.onnx
MyDrive/grid-bot/model_output/norm_stats.json          -> backend/model/norm_stats.json
```

## Natijani baholash bo'yicha eslatma

- Test accuracy tasodifiy taxmindan (33%, 3 klass) sezilarli yuqori bo'lishi kerak.
- Agar Buy/Sell precision past chiqsa — `data_pipeline/config.py` dagi
  `TP_ATR_MULT`/`SL_ATR_MULT`/`LOOKAHEAD_BARS` qiymatlarini o'zgartirib,
  Bosqich 1 (label_generator.py, render_charts.py) ni qayta ishga tushirish mumkin.
