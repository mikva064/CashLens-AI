"""Подключение AI-модуля cashlens_ai к сервису.

Сервис работает с одним интерфейсом:

    analyzer.load()                                   # один раз, до приёма задач
    analyzer.analyze(video, output_dir, deadline_sec) -> cashlens_ai.AnalysisResult

CASHLENS_ANALYZER:
  model — локальная Qwen-модель (src/cashlens_ai), нужны веса и проверенные правила;
  mock  — демо-заглушка для разработки интерфейса, ничего не распознаёт.
"""

import os
from pathlib import Path
from typing import Protocol

from cashlens_ai.contracts import AnalysisResult


class Analyzer(Protocol):
    name: str

    def load(self) -> None: ...

    def analyze(self, video: Path, output_dir: Path, deadline_sec: float) -> AnalysisResult: ...


def get_analyzer() -> Analyzer:
    name = os.getenv("CASHLENS_ANALYZER", "mock")
    if name == "mock":
        from analyzer.mock import MockAnalyzer
        return MockAnalyzer()
    if name == "model":
        from analyzer.qwen import QwenAnalyzer
        return QwenAnalyzer()
    raise ValueError(f"Неизвестный анализатор: {name}")
