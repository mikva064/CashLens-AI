"""Демо-заглушка: ответ в формате cashlens_ai.AnalysisResult без распознавания.

Нужна, чтобы сервис и интерфейс работали без GPU, весов и заполненных правил.
В ответе явно помечено, что это демо.
"""

import hashlib
import os
import time
from pathlib import Path

from cashlens_ai.contracts import AnalysisResult, EvidenceFrame, RuleSource, Violation
from cashlens_ai.video import get_frame_at, probe_video

DEMO_SOURCE = RuleSource(
    document="ДЕМО-РЕЖИМ: модель не подключена, источник не выбран",
    section="пункт не выбран",
    url="https://adilet.zan.kz/rus/docs/V1900019680",
)


class MockAnalyzer:
    name = "mock"

    def load(self) -> None:
        pass

    def analyze(self, video: Path, output_dir: Path, deadline_sec: float) -> AnalysisResult:
        time.sleep(float(os.getenv("CASHLENS_MOCK_DELAY", "2")))
        output_dir.mkdir(parents=True, exist_ok=False)

        # CASHLENS_MOCK_VERDICT=violation|clean задаёт ответ явно; по умолчанию он зависит
        # от содержимого файла, чтобы в интерфейсе можно было увидеть оба варианта.
        mode = os.getenv("CASHLENS_MOCK_VERDICT", "auto")
        if mode == "auto":
            with open(video, "rb") as f:
                mode = "violation" if hashlib.md5(f.read(1 << 20)).digest()[0] % 2 else "clean"
        if mode == "clean":
            return AnalysisResult(verdict="no_violation", violations=[])

        info = probe_video(video)
        anchor = get_frame_at(video, info.duration_sec * 0.5)
        evidence = []
        for i, offset in enumerate((0.0, -0.5, 0.5), start=1):
            frame = anchor if offset == 0 else get_frame_at(video, anchor.timestamp_sec + offset)
            path = output_dir / f"event_001_{i}.jpg"
            frame.image.save(path, format="JPEG", quality=92)
            evidence.append(EvidenceFrame(timestamp_sec=frame.timestamp_sec, image_path=path))
        return AnalysisResult(verdict="violation", violations=[Violation(
            type="Демо-событие",
            timecode_sec=anchor.timestamp_sec,
            description="Тестовое событие от заглушки: модель не подключена, видео не анализировалось.",
            rule_id="demo_event",
            source=DEMO_SOURCE,
            evidence_frames=evidence,
        )])
