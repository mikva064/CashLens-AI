"""Один фоновый воркер: задачи обрабатываются строго по очереди.

На одной GPU параллельный запуск модели только замедляет работу и рискует OOM.
Модель загружается один раз при старте воркера.
"""

import logging
import threading
import time
import traceback

from analyzer import get_analyzer
from app import db, media
from app.config import JOBS_DIR, JOB_TIMEOUT_SEC

log = logging.getLogger("cashlens.worker")

# Состояние анализатора для /health и интерфейса.
state = {"analyzer": None, "ready": False, "error": None}


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
    try:
        analyzer = get_analyzer()
        state["analyzer"] = analyzer.name
        analyzer.load()
        state["ready"] = True
        log.info("analyzer %s loaded", analyzer.name)
    except Exception as exc:
        # Без анализатора задачи не обрабатываются (и не получают ложный no_violation).
        state["error"] = f"{type(exc).__name__}: {exc}"
        log.error("analyzer failed to load\n%s", traceback.format_exc())
        analyzer = None

    while not stop.is_set():
        job_id = db.claim_next()
        if job_id is None:
            stop.wait(0.5)
            continue
        if analyzer is None:
            db.fail(job_id, f"Анализатор не загружен: {state['error']}")
            continue
        try:
            _process(job_id, analyzer)
        except Exception as exc:  # любая ошибка анализа -> failed, сервис живёт дальше
            log.error("job %s failed\n%s", job_id, traceback.format_exc())
            db.fail(job_id, f"{type(exc).__name__}: {exc}")


def start() -> threading.Event:
    stop = threading.Event()
    threading.Thread(target=_loop, args=(stop,), name="cashlens-worker", daemon=True).start()
    return stop
