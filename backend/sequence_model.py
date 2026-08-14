"""Vaqt ketma-ketligini (oxirgi N bar) o'rganadigan LSTM model -- "video" g'oyasining
samarali versiyasi: piksel o'rniga har bar uchun ATR-normallashtirilgan
(tana, fitilar, diapazon) xususiyatlarini ketma-ket beradi, model esa vaqt
bo'yicha o'zgarishni (LSTM) o'rganadi. Bundan tashqari joriy nuqtadagi statik
indikatorlar (RSI, MACD va h.k.) ham qo'shilib, yakuniy qarorga ta'sir qiladi.

Training (`training/train_sequence_model.py`) va inference (`custom_model.py`)
shu modul orqali BIR XIL arxitektura va sequence qurish mantig'idan foydalanadi.
"""
import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 30
BAR_FEATURE_COUNT = 5  # ret, body, upper_wick, lower_wick, range


def build_bar_sequence(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray, atr: float, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Oxirgi `seq_len` bar uchun (seq_len, 5) shaklidagi ATR-normallashtirilgan
    xususiyat ketma-ketligini quradi. Tarix yetarli bo'lmasa boshidan nol bilan
    to'ldiriladi."""
    atr = atr if atr and atr > 0 else 1e-9
    n = len(closes)
    take = min(seq_len, n)

    o, h, l, c = opens[-take:], highs[-take:], lows[-take:], closes[-take:]
    prev_c = np.roll(c, 1)
    prev_c[0] = o[0]

    ret = (c - prev_c) / atr
    body = (c - o) / atr
    upper_wick = (h - np.maximum(o, c)) / atr
    lower_wick = (np.minimum(o, c) - l) / atr
    rng = (h - l) / atr

    seq = np.stack([ret, body, upper_wick, lower_wick, rng], axis=1).astype(np.float32)

    if take < seq_len:
        pad = np.zeros((seq_len - take, BAR_FEATURE_COUNT), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)
    return seq


class SequenceModel(nn.Module):
    def __init__(self, static_dim: int, num_classes: int = 3, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=BAR_FEATURE_COUNT, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, num_classes),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq)
        combined = torch.cat([h_n[-1], static], dim=1)
        return self.head(combined)
