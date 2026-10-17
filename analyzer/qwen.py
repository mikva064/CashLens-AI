"""Локальная модель Qwen3-VL через пайплайн cashlens_ai (см. docs/integration.md)."""

import dataclasses
import os
from pathlib import Path

from app.config import ROOT
from cashlens_ai.backend import LocalQwenBackend
from cashlens_ai.contracts import AnalysisResult
from cashlens_ai.pipeline import AnalyzerConfig, VideoAnalyzer
from cashlens_ai.rules import load_rules

# Запас между бюджетом модели и лимитом задачи: кадры в base64, ответ, сеть.
SAFETY_SEC = 20.0


class QwenAnalyzer:
    name = "model"

    def __init__(self) -> None:
        self.model_dir = Path(os.getenv("CASHLENS_MODEL_DIR", ROOT / "local" / "models" / "qwen3-vl-2b"))
        self.rules_path = Path(os.getenv("CASHLENS_RULES", ROOT / "config" / "rules.json"))
        self._analyzer: VideoAnalyzer | None = None

    def load(self) -> None:
        # Правила проверяются до загрузки весов: пустой реестр блокирует анализ.
        rules = load_rules(self.rules_path)
        backend = LocalQwenBackend(self.model_dir)
        backend.load()
        self._analyzer = VideoAnalyzer(backend, rules)

    def analyze(self, video: Path, output_dir: Path, deadline_sec: float) -> AnalysisResult:
        assert self._analyzer is not None, "load() не вызван"
        base = self._analyzer.config
        budget = min(base.time_budget_sec, deadline_sec - SAFETY_SEC)
        self._analyzer.config = dataclasses.replace(base, time_budget_sec=max(budget, 1.0))
        try:
            return self._analyzer.analyze(video, output_dir)
        finally:
            self._analyzer.config = base
