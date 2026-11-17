"""Local Qwen image analysis through Ollama; independent of final consensus."""
import base64
import json
import math
import threading
import bridge_client
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import OLLAMA_URL, QWEN_MODEL, QWEN_TIMEOUT
from gemini_vision import _format_context

_CONFIG_LOCK = threading.RLock()


def configure(model, timeout):
    global QWEN_MODEL, QWEN_TIMEOUT
    with _CONFIG_LOCK:
        QWEN_MODEL, QWEN_TIMEOUT = model, timeout

PROMPT_VERSION = "qwen-chart-v1"
_PROMPT = """Analyze the candlestick chart and provided MT5 indicators. Reply in Uzbek.
Return JSON with signal (Buy, Sell, or Hold), confidence (number between 0 and 1),
trend (short string), reasoning (1-2 sentences). Express uncertainty when needed.
Do not invent prices. This is a separate experimental opinion, not an order."""
_SCHEMA = {
    "type": "object",
    "properties": {
        "signal": {"type": "string", "enum": ["Buy", "Sell", "Hold"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "trend": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["signal", "confidence", "trend", "reasoning"],
}


def analyze_image(image_bytes: bytes, market_context: dict | None = None, *, model=None, timeout=None) -> dict:
    with _CONFIG_LOCK:
        model = QWEN_MODEL if model is None else model
        timeout = QWEN_TIMEOUT if timeout is None else timeout
    context = _format_context(market_context) if market_context else "MT5 context unavailable."
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{_PROMPT}\n\n{context}",
                      "images": [base64.b64encode(image_bytes).decode("ascii")]}],
        "stream": False, "think": False, "format": _SCHEMA,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 400},
        "keep_alive": 0,
    }
    req = Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        if bridge_client.enabled():
            result = bridge_client.request("ollama_chat", {"payload": payload, "timeout": timeout}, timeout=timeout + 30)
        else:
            with urlopen(req, timeout=timeout) as response:
                result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP {exc.code}: model mavjudligini tekshiring ({model})") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Ollama ulanmagan yoki javob muddati tugadi ({model})") from exc
    try:
        parsed = json.loads(result["message"]["content"])
        signal = parsed["signal"]
        confidence = float(parsed["confidence"])
        trend, reasoning = parsed["trend"], parsed["reasoning"]
        if signal not in ("Buy", "Sell", "Hold") or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid signal or confidence")
        if not isinstance(trend, str) or not isinstance(reasoning, str):
            raise ValueError("invalid explanation")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Qwen javobi kutilgan formatda emas") from exc
    return {"source": "qwen", "model": model, "prompt_version": PROMPT_VERSION,
            "signal": signal, "confidence": confidence, "trend": trend, "reasoning": reasoning}
