"""Точка подключения анализатора. Сервис вызывает только get_analyzer()."""

import os
from typing import Callable

AnalyzeFn = Callable[[str, float], dict]


def get_analyzer() -> AnalyzeFn:
    name = os.getenv("CASHLENS_ANALYZER", "mock")
    if name == "mock":
        from analyzer.mock import analyze_video
        return analyze_video
    if name == "model":
        # Модуль человека 1: analyzer/model.py с функцией analyze_video(path, deadline_sec).
        from analyzer.model import analyze_video
        return analyze_video
    raise ValueError(f"Неизвестный анализатор: {name}")
