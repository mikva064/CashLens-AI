"""Заглушка анализатора: детерминированный ответ по формату контракта.

Нужна, чтобы сервис и интерфейс разрабатывались до готовности модели.
Ничего не распознаёт.
"""

import hashlib
import os
import time

from app.media import probe_duration


def analyze_video(path: str, deadline_sec: float) -> dict:
    time.sleep(float(os.getenv("CASHLENS_MOCK_DELAY", "2")))
    duration = probe_duration(path) or 30.0

    # CASHLENS_MOCK_VERDICT=violation|clean задаёт ответ явно; по умолчанию он зависит
    # от содержимого файла, чтобы в интерфейсе можно было увидеть оба варианта.
    mode = os.getenv("CASHLENS_MOCK_VERDICT", "auto")
    if mode == "auto":
        with open(path, "rb") as f:
            mode = "violation" if hashlib.md5(f.read(1 << 20)).digest()[0] % 2 else "clean"
    if mode == "clean":
        return {"events": []}

    key = round(duration * 0.5, 1)
    return {
        "events": [
            {
                "type": "example_violation",
                "start_sec": max(0.0, key - 2),
                "end_sec": min(duration, key + 2),
                "key_sec": key,
                "confidence": 0.5,
                "description": "Тестовое событие от заглушки анализатора.",
            }
        ]
    }
