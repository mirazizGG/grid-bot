"""train_colab.ipynb faylini generatsiya qiladi (JSON qo'lda yozishdan ko'ra ishonchliroq)."""
import json

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines]}

cells = []

cells.append(md(
    "# Grid-Bot: Chart Signal Model (XAUUSD M15)",
    "",
    "Bu notebook `dataset/` papkasidagi candlestick rasmlar asosida Buy/Sell/Hold",
    "klassifikatsiyasi va TP/SL regressiyasini o'rgatadi (ResNet18 transfer learning,",
    "ikki boshli model). Runtime > Change runtime type > **GPU (T4)** tanlang.",
))

cells.append(md(
    "## 1. Google Drive ulash va dataset arxivini ochish",
    "",
    "Oldindan `dataset_grid_bot.zip` faylini Google Drive'ingizga yuklab qo'ying",
    "(masalan `MyDrive/grid-bot/dataset_grid_bot.zip`) va quyidagi yo'lni moslang.",
))

cells.append(code(
    "from google.colab import drive",
    "drive.mount('/content/drive')",
))

cells.append(code(
    "DRIVE_ZIP_PATH = '/content/drive/MyDrive/grid-bot/dataset_grid_bot.zip'  # o'zgartiring",
    "DATASET_DIR = '/content/dataset'",
    "",
    "!mkdir -p {DATASET_DIR}",
    "!unzip -q -o \"{DRIVE_ZIP_PATH}\" -d {DATASET_DIR}",
    "!find {DATASET_DIR} -name '*.png' | wc -l",
))

cells.append(md("## 2. Kutubxonalar"))

cells.append(code(
    "!pip install -q onnx onnxruntime",
    "",
    "import os",
    "import json",
    "",
    "import numpy as np",
    "import pandas as pd",
    "import torch",
    "import torch.nn as nn",
    "from torch.utils.data import Dataset, DataLoader",
    "from torchvision import models, transforms",
    "from PIL import Image",
    "from sklearn.metrics import classification_report, confusion_matrix",
    "",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "print('device:', device)",
))

cells.append(md(
    "## 3. Manifest yuklash va normalizatsiya statistikasi",
    "",
    "`tp_price`/`sl_price` xom narx birliklarida (turli symbol/timeframe uchun",
    "shkala farq qiladi), shuning uchun regressiya boshini train split statistikasi",
    "(mean/std) bilan normallashtiramiz. Bu qiymatlar keyinchalik inference paytida",
    "ham kerak bo'ladi -> `norm_stats.json` ga saqlanadi.",
))

cells.append(code(
    "MANIFEST_PATH = os.path.join(DATASET_DIR, 'manifest.csv')",
    "df = pd.read_csv(MANIFEST_PATH)",
    "",
    "# manifest.csv image_path lokal mashinadagi to'liq yo'l bilan yozilgan edi;",
    "# faqat 'dataset/...' dan keyingi qismini olib, Colab yo'liga moslaymiz.",
    "def fix_path(p):",
    "    p = p.replace('\\\\', '/')",
    "    idx = p.find('dataset/images/')",
    "    rel = p[idx + len('dataset/'):] if idx != -1 else p",
    "    return os.path.join(DATASET_DIR, rel)",
    "",
    "df['image_path'] = df['image_path'].apply(fix_path)",
    "assert os.path.exists(df['image_path'].iloc[0]), df['image_path'].iloc[0]",
    "",
    "LABEL_MAP = {'Buy': 0, 'Sell': 1, 'Hold': 2}",
    "df['label_id'] = df['label'].map(LABEL_MAP)",
    "",
    "train_df = df[df['split'] == 'train'].reset_index(drop=True)",
    "val_df = df[df['split'] == 'val'].reset_index(drop=True)",
    "test_df = df[df['split'] == 'test'].reset_index(drop=True)",
    "print(len(train_df), len(val_df), len(test_df))",
    "",
    "tp_mean, tp_std = train_df['tp_price'].mean(), train_df['tp_price'].std()",
    "sl_mean, sl_std = train_df['sl_price'].mean(), train_df['sl_price'].std()",
    "norm_stats = {'tp_mean': float(tp_mean), 'tp_std': float(tp_std),",
    "              'sl_mean': float(sl_mean), 'sl_std': float(sl_std)}",
    "norm_stats",
))

cells.append(md("## 4. Dataset / DataLoader"))

cells.append(code(
    "IMG_SIZE = 224",
    "",
    "train_tf = transforms.Compose([",
    "    transforms.Resize((IMG_SIZE, IMG_SIZE)),",
    "    transforms.ColorJitter(brightness=0.1, contrast=0.1),",
    "    transforms.ToTensor(),",
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),",
    "])",
    "eval_tf = transforms.Compose([",
    "    transforms.Resize((IMG_SIZE, IMG_SIZE)),",
    "    transforms.ToTensor(),",
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),",
    "])",
    "",
    "class ChartDataset(Dataset):",
    "    def __init__(self, frame, transform):",
    "        self.frame = frame",
    "        self.transform = transform",
    "",
    "    def __len__(self):",
    "        return len(self.frame)",
    "",
    "    def __getitem__(self, idx):",
    "        row = self.frame.iloc[idx]",
    "        img = Image.open(row['image_path']).convert('RGB')",
    "        img = self.transform(img)",
    "        label = int(row['label_id'])",
    "        tp_norm = np.clip((row['tp_price'] - tp_mean) / tp_std, -4, 4)",
    "        sl_norm = np.clip((row['sl_price'] - sl_mean) / sl_std, -4, 4)",
    "        target = torch.tensor([tp_norm, sl_norm], dtype=torch.float32)",
    "        return img, label, target",
    "",
    "BATCH_SIZE = 64",
    "train_ds = ChartDataset(train_df, train_tf)",
    "val_ds = ChartDataset(val_df, eval_tf)",
    "test_ds = ChartDataset(test_df, eval_tf)",
    "",
    "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)",
    "val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)",
    "test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)",
))

cells.append(md(
    "## 5. Model: ResNet18 backbone + ikki boshli chiqish",
    "",
    "- Classification head -> Buy/Sell/Hold (3 klass)",
    "- Regression head -> normallashtirilgan (TP, SL) masofa",
))

cells.append(code(
    "class ChartSignalModel(nn.Module):",
    "    def __init__(self, num_classes=3, dropout=0.5):",
    "        super().__init__()",
    "        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)",
    "        in_features = backbone.fc.in_features",
    "        backbone.fc = nn.Identity()",
    "        self.backbone = backbone",
    "        self.dropout = nn.Dropout(dropout)",
    "        self.classifier = nn.Linear(in_features, num_classes)",
    "        self.regressor = nn.Linear(in_features, 2)  # [tp_norm, sl_norm]",
    "",
    "    def forward(self, x):",
    "        features = self.dropout(self.backbone(x))",
    "        return self.classifier(features), self.regressor(features)",
    "",
    "model = ChartSignalModel().to(device)",
    "",
    "# Dataset kichik (~3000 rasm) bo'lgani uchun backbone'ning dastlabki",
    "# qatlamlarini muzlatib, faqat yuqori qatlamlar + head'larni o'qitamiz --",
    "# bu overfitting xavfini kamaytiradi.",
    "for name, param in model.backbone.named_parameters():",
    "    if not (name.startswith('layer4') or name.startswith('layer3')):",
    "        param.requires_grad = False",
))

cells.append(md(
    "## 6. Class weight, loss, optimizer",
    "",
    "Hold/Buy/Sell taqsimoti biroz nomutanosib (Hold kamroq), shuning uchun",
    "class weight bilan cross-entropy'ni balanslaymiz.",
))

cells.append(code(
    "class_counts = train_df['label_id'].value_counts().sort_index()",
    "class_weights = torch.tensor(",
    "    (class_counts.sum() / (len(class_counts) * class_counts)).values,",
    "    dtype=torch.float32,",
    ").to(device)",
    "print('class_weights (Buy,Sell,Hold):', class_weights)",
    "",
    "cls_criterion = nn.CrossEntropyLoss(weight=class_weights)",
    "reg_criterion = nn.MSELoss()",
    "REG_LOSS_WEIGHT = 0.3",
    "",
    "trainable_params = [p for p in model.parameters() if p.requires_grad]",
    "optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-4)",
    "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)",
))

cells.append(md("## 7. O'qitish sikli"))

cells.append(code(
    "def run_epoch(loader, train_mode):",
    "    model.train() if train_mode else model.eval()",
    "    total_loss, correct, n = 0.0, 0, 0",
    "    with torch.set_grad_enabled(train_mode):",
    "        for imgs, labels, targets in loader:",
    "            imgs, labels, targets = imgs.to(device), labels.to(device), targets.to(device)",
    "            if train_mode:",
    "                optimizer.zero_grad()",
    "            logits, reg_out = model(imgs)",
    "            loss = cls_criterion(logits, labels) + REG_LOSS_WEIGHT * reg_criterion(reg_out, targets)",
    "            if train_mode:",
    "                loss.backward()",
    "                optimizer.step()",
    "            total_loss += loss.item() * imgs.size(0)",
    "            correct += (logits.argmax(1) == labels).sum().item()",
    "            n += imgs.size(0)",
    "    return total_loss / n, correct / n",
    "",
    "EPOCHS = 25",
    "EARLY_STOP_PATIENCE = 5",
    "best_val_acc = 0.0",
    "epochs_without_improvement = 0",
    "for epoch in range(1, EPOCHS + 1):",
    "    train_loss, train_acc = run_epoch(train_loader, True)",
    "    val_loss, val_acc = run_epoch(val_loader, False)",
    "    scheduler.step(val_acc)",
    "    print(f'epoch {epoch:02d} | train_loss {train_loss:.4f} acc {train_acc:.3f} | val_loss {val_loss:.4f} acc {val_acc:.3f}')",
    "    if val_acc > best_val_acc:",
    "        best_val_acc = val_acc",
    "        epochs_without_improvement = 0",
    "        torch.save(model.state_dict(), 'best_model.pt')",
    "    else:",
    "        epochs_without_improvement += 1",
    "        if epochs_without_improvement >= EARLY_STOP_PATIENCE:",
    "            print(f'Erta to\\'xtatildi: {EARLY_STOP_PATIENCE} epoch val_acc yaxshilanmadi')",
    "            break",
    "print('best val acc:', best_val_acc)",
))

cells.append(md("## 8. Test to'plamida baholash"))

cells.append(code(
    "model.load_state_dict(torch.load('best_model.pt'))",
    "model.eval()",
    "",
    "all_preds, all_labels = [], []",
    "with torch.no_grad():",
    "    for imgs, labels, targets in test_loader:",
    "        imgs = imgs.to(device)",
    "        logits, _ = model(imgs)",
    "        preds = logits.argmax(1).cpu().numpy()",
    "        all_preds.extend(preds)",
    "        all_labels.extend(labels.numpy())",
    "",
    "print(classification_report(all_labels, all_preds, target_names=['Buy', 'Sell', 'Hold']))",
    "print(confusion_matrix(all_labels, all_preds))",
))

cells.append(md(
    "## 9. ONNX'ga eksport",
    "",
    "Backend'da GPU shart bo'lmasligi uchun ONNX Runtime bilan CPU'da tez",
    "inference qilinadi.",
))

cells.append(code(
    "dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)",
    "torch.onnx.export(",
    "    model, dummy, 'chart_signal_model.onnx',",
    "    input_names=['image'], output_names=['class_logits', 'tp_sl_norm'],",
    "    dynamic_axes={'image': {0: 'batch'}, 'class_logits': {0: 'batch'}, 'tp_sl_norm': {0: 'batch'}},",
    "    opset_version=17,",
    ")",
    "",
    "with open('norm_stats.json', 'w') as f:",
    "    json.dump({**norm_stats, 'label_map': LABEL_MAP, 'img_size': IMG_SIZE}, f, indent=2)",
    "",
    "print('Tayyor: chart_signal_model.onnx, norm_stats.json')",
))

cells.append(md(
    "## 10. Natijalarni Drive'ga saqlash",
    "",
    "`chart_signal_model.onnx` va `norm_stats.json` fayllarini keyinchalik",
    "`backend/model/` papkasiga qo'yish uchun Drive'ga (yoki to'g'ridan-to'g'ri",
    "kompyuteringizga) yuklab oling.",
))

cells.append(code(
    "!mkdir -p /content/drive/MyDrive/grid-bot/model_output",
    "!cp chart_signal_model.onnx norm_stats.json /content/drive/MyDrive/grid-bot/model_output/",
    "print('Drive/grid-bot/model_output/ ga nusxalandi')",
))

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "train_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("train_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("train_colab.ipynb yaratildi")
