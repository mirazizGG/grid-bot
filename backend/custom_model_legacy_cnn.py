"""ONNX Runtime orqali o'zimiz o'qitgan chart signal modelidan inference qilish."""
import json

import numpy as np
import onnxruntime as ort
from PIL import Image

from config import ONNX_MODEL_PATH, NORM_STATS_PATH

with open(NORM_STATS_PATH) as f:
    _NORM_STATS = json.load(f)

_IMG_SIZE = _NORM_STATS["img_size"]
_ID_TO_LABEL = {v: k for k, v in _NORM_STATS["label_map"].items()}
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((_IMG_SIZE, _IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis, ...].astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def analyze_image(image: Image.Image) -> dict:
    x = _preprocess(image)
    class_logits, tp_sl_norm = _session.run(None, {_input_name: x})

    probs = _softmax(class_logits[0])
    class_id = int(np.argmax(probs))
    label = _ID_TO_LABEL[class_id]
    confidence = float(probs[class_id])

    tp_norm, sl_norm = tp_sl_norm[0]
    tp_price = float(tp_norm) * _NORM_STATS["tp_std"] + _NORM_STATS["tp_mean"]
    sl_price = float(sl_norm) * _NORM_STATS["sl_std"] + _NORM_STATS["sl_mean"]

    return {
        "source": "custom_model",
        "signal": label,
        "confidence": confidence,
        "class_probs": {_ID_TO_LABEL[i]: float(p) for i, p in enumerate(probs)},
        "tp_distance": max(tp_price, 0.0),
        "sl_distance": max(sl_price, 0.0),
    }
