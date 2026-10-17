"""Локальная модель Qwen3-VL через пайплайн cashlens_ai (см. docs/integration.md)."""

import dataclasses
import os
from pathlib import Path

from app.config import ROOT
from cashlens_ai.backend import LocalQwenBackend
from cashlens_ai.contracts import AnalysisResult
from cashlens_ai.errors import AnalysisTimeoutError
from cashlens_ai.pipeline import AnalyzerConfig, VideoAnalyzer
from cashlens_ai.rules import load_rules

# Запас между бюджетом модели и лимитом задачи: кадры в base64, ответ, сеть.
SAFETY_SEC = 20.0
PROFILE_DEFAULTS = {
    "balanced": {"max_edge": 384, "refine_max_edge": 640,
                 "max_pixels": 131072, "max_refine_frames": 8},
    "quality": {"max_edge": 448, "refine_max_edge": 896,
                "max_pixels": 262144, "max_refine_frames": 12},
}


class QwenAnalyzer:
    name = "model"

    def __init__(self) -> None:
        self.model_dir = self._path("CASHLENS_MODEL_DIR", "local/models/qwen3-vl-2b")
        self.rules_path = self._path("CASHLENS_RULES", "config/rules.json")
        self._analyzer: VideoAnalyzer | None = None
        self._backend: LocalQwenBackend | None = None
        self._rules = []
        self.profile = os.getenv("CASHLENS_PERFORMANCE_PROFILE", "balanced").strip().lower()
        self._performance = {}

    @staticmethod
    def _path(name: str, default: str) -> Path:
        value = Path(os.getenv(name, default))
        return value if value.is_absolute() else ROOT / value

    def capabilities(self) -> dict:
        return {
            "model_name": os.getenv("CASHLENS_MODEL_NAME", "Qwen3-VL-2B-Instruct"),
            "rule_count": len(self._rules),
            "rules": [rule.model_dump(mode="json", exclude={"verified"}) for rule in self._rules],
            "performance": dict(self._performance),
            "limitations": (
                "Модель ищет видимые признаки только перечисленных событий. "
                "Краткие действия и скрытые объекты могут быть пропущены. "
                "Результат требует проверки человеком и не устанавливает факт правонарушения. "
                "Отсутствие найденных событий не подтверждает соблюдение всех требований законодательства."
            ),
        }

    def load(self) -> None:
        if self.profile not in PROFILE_DEFAULTS:
            raise ValueError("CASHLENS_PERFORMANCE_PROFILE must be balanced or quality")
        defaults = PROFILE_DEFAULTS[self.profile]
        # Правила проверяются до загрузки весов: пустой реестр блокирует анализ.
        self._rules = load_rules(self.rules_path)
        config = AnalyzerConfig(
            sample_fps=float(os.getenv("CASHLENS_SAMPLE_FPS", "1")),
            window_sec=float(os.getenv("CASHLENS_WINDOW_SEC", "24")),
            overlap_sec=float(os.getenv("CASHLENS_OVERLAP_SEC", "4")),
            max_edge=int(os.getenv("CASHLENS_MAX_EDGE", str(defaults["max_edge"]))),
            refine_max_edge=int(os.getenv("CASHLENS_REFINE_MAX_EDGE", str(defaults["refine_max_edge"]))),
            max_refine_frames=int(os.getenv("CASHLENS_MAX_REFINE_FRAMES", str(defaults["max_refine_frames"]))),
        )
        max_pixels = int(os.getenv("CASHLENS_MAX_PIXELS", str(defaults["max_pixels"])))
        gpu_duty_cycle = float(os.getenv("CASHLENS_GPU_DUTY_CYCLE", "0.75"))
        gpu_memory_fraction = float(os.getenv("CASHLENS_GPU_MEMORY_FRACTION", "0.80"))
        self._backend = LocalQwenBackend(
            self.model_dir,
            quantize_4bit=os.getenv("CASHLENS_QUANTIZE_4BIT", "1") == "1",
            max_new_tokens=int(os.getenv("CASHLENS_MAX_NEW_TOKENS", "384")),
            max_pixels=max_pixels,
            gpu_duty_cycle=gpu_duty_cycle,
            gpu_memory_fraction=gpu_memory_fraction,
        )
        try:
            self._backend.load()
            self._analyzer = VideoAnalyzer(self._backend, self._rules, config=config)
            self._performance = {
                "profile": self.profile, "sample_fps": config.sample_fps,
                "window_sec": config.window_sec, "overlap_sec": config.overlap_sec,
                "max_edge": config.max_edge, "refine_max_edge": config.refine_max_edge,
                "max_refine_frames": config.max_refine_frames, "max_pixels": max_pixels,
                "gpu_duty_cycle": gpu_duty_cycle, "gpu_memory_fraction": gpu_memory_fraction,
            }
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._analyzer = None
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def analyze(self, video: Path, output_dir: Path, deadline_sec: float) -> AnalysisResult:
        if self._analyzer is None:
            raise RuntimeError("Локальная модель ещё не загружена")
        base = self._analyzer.config
        try:
            budget = min(base.time_budget_sec, deadline_sec - SAFETY_SEC)
            if budget <= 0:
                raise AnalysisTimeoutError("Insufficient remaining job budget")
            self._analyzer.config = dataclasses.replace(base, time_budget_sec=budget)
            return self._analyzer.analyze(video, output_dir)
        finally:
            self._analyzer.config = base
            if self._backend is not None:
                self._backend.release_unused_memory()
