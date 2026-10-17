"""Один фоновый воркер: задачи обрабатываются строго по очереди.

На одной GPU параллельный запуск модели только замедляет работу и рискует OOM.
Модель загружается один раз при старте воркера.
"""

import logging
import os
import threading
import time
import traceback

from analyzer import get_analyzer
from app import db, media
from app.config import JOBS_DIR, JOB_TIMEOUT_SEC
from cashlens_ai.errors import (AnalysisTimeoutError, InvalidModelOutputError,
                                ModelUnavailableError, RuleValidationError, VideoReadError)

log = logging.getLogger("cashlens.worker")

# Состояние анализатора для /health и интерфейса.
state = {"analyzer": None, "ready": False, "error": None, "status": "stopped",
         "model_name": None, "rule_count": 0, "rules": [], "limitations": "",
         "performance": {}}
_state_lock = threading.Lock()
_thread: threading.Thread | None = None


def _update(**values) -> None:
    with _state_lock:
        state.update(values)


def snapshot() -> dict:
    with _state_lock:
        return dict(state)


def _public_error(exc: Exception, *, loading: bool = False) -> str:
    """Keep diagnostics in local logs; responses must not expose paths or stack traces."""
    if isinstance(exc, RuleValidationError):
        return "Реестр проверенных правил не настроен. Проверьте конфигурацию сервиса."
    if isinstance(exc, AnalysisTimeoutError):
        return "Обработка не уложилась в лимит времени. Полный результат не получен."
    if isinstance(exc, InvalidModelOutputError):
        return "Не удалось проверить ответ модели. Вердикт не сформирован; попробуйте повторить анализ."
    if isinstance(exc, VideoReadError):
        return "Не удалось прочитать видео. Проверьте файл и повторите загрузку."
    if isinstance(exc, ModelUnavailableError):
        return ("Локальная модель недоступна. Проверьте наличие весов, CUDA и свободную видеопамять. "
                "Подробности доступны в журнале сервиса.")
    if loading:
        return "Не удалось запустить локальную модель. Подробности доступны в журнале сервиса."
    return "Ошибка обработки видео. Вердикт не сформирован; подробности доступны в журнале сервиса."


def _process(job_id: str, analyzer) -> None:
    job = db.get(job_id)
    job_dir = JOBS_DIR / job_id
    src = job_dir / "video.mov"
    t0 = time.time()
    # Отсчёт лимита — от загрузки, а не от начала обработки: жюри ждёт с момента POST.
    deadline = JOB_TIMEOUT_SEC - (t0 - job["created_at"])
    if deadline <= 0:
        db.fail(job_id, "Превышено время ожидания в очереди")
        return
    # mp4 для плеера нужен только интерфейсу — конвертируем параллельно, результат его не ждёт.
    threading.Thread(target=media.to_browser_mp4, args=(src, job_dir / "preview.mp4"),
                     daemon=True).start()
    result = analyzer.analyze(src, job_dir / "ai", deadline)
    if time.time() - job["created_at"] >= JOB_TIMEOUT_SEC:
        raise AnalysisTimeoutError("Job deadline exceeded after model generation")
    db.finish(job_id, {
        "analysis": result.model_dump(mode="json"),
        "analyzer": analyzer.name,
        "duration_sec": media.probe_duration(src),
        "timings": {"queue_sec": round(t0 - job["created_at"], 2),
                    "processing_sec": round(time.time() - t0, 2)},
    })
    log.info("job %s done: %s, %d violations, %.1fs",
             job_id, result.verdict, len(result.violations), time.time() - t0)


def _loop(stop: threading.Event) -> None:
    analyzer = None
    try:
        analyzer = get_analyzer()
        _update(analyzer=analyzer.name)
        analyzer.load()
        capabilities = getattr(analyzer, "capabilities", lambda: {})()
        _update(**capabilities, ready=True, status="ready", error=None)
        log.info("analyzer %s loaded", analyzer.name)
    except Exception as exc:
        # Без анализатора задачи не обрабатываются (и не получают ложный no_violation).
        _update(ready=False, status="failed", error=_public_error(exc, loading=True))
        log.error("analyzer failed to load\n%s", traceback.format_exc())
    try:
        while not stop.is_set():
            job_id = db.claim_next()
            if job_id is None:
                stop.wait(0.2)
                continue
            if not snapshot()["ready"]:
                db.fail(job_id, snapshot()["error"] or "Локальная модель недоступна")
                continue
            try:
                _process(job_id, analyzer)
            except Exception as exc:  # любая ошибка анализа -> failed, сервис живёт дальше
                # Chained model-validation errors can include scene descriptions
                # in Pydantic input_value. Keep video context out of generic logs.
                public_error = _public_error(exc)
                log.error("analysis failed (%s): %s", type(exc).__name__, public_error)
                db.fail(job_id, public_error)
    finally:
        # The worker owns the GPU. Release it in this thread after current inference exits.
        close = getattr(analyzer, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                log.exception("Could not release analyzer resources")
        _update(ready=False, status="stopped")


def start() -> threading.Event:
    global _thread
    if _thread is not None and _thread.is_alive():
        raise RuntimeError("The previous analyzer worker is still running")
    _update(analyzer=os.getenv("CASHLENS_ANALYZER", "model").strip().lower(),
            ready=False, error=None, status="loading", model_name=None,
            rule_count=0, rules=[], limitations="", performance={})
    stop = threading.Event()
    _thread = threading.Thread(target=_loop, args=(stop,), name="cashlens-worker", daemon=True)
    _thread.start()
    return stop


def shutdown(stop: threading.Event) -> None:
    stop.set()
    if _thread is not None:
        _thread.join(timeout=2)
