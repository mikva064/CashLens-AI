"""Один фоновый воркер: задачи обрабатываются строго по очереди.

На одной GPU параллельный запуск модели только замедляет работу и рискует OOM.
"""

import logging
import threading
import time
import traceback

from analyzer import get_analyzer
from app import db, media, rules
from app.config import JOBS_DIR, JOB_TIMEOUT_SEC

log = logging.getLogger("cashlens.worker")

# Значения вердикта — по схеме API из ТЗ (раздел 10).
VERDICT_OK = "no_violation"
VERDICT_VIOLATION = "violation"


def _build_result(job_id: str, raw: dict, duration: float | None) -> dict:
    job_dir = JOBS_DIR / job_id
    src = job_dir / "video.mov"
    violations = []
    for i, ev in enumerate(raw.get("events", []), start=1):
        start = float(ev["start_sec"])
        end = float(ev.get("end_sec", start))
        key = round(float(ev.get("key_sec", (start + end) / 2)), 2)
        # Первый кадр — ровно на таймкоде; затем «до» и «после» для последовательности действия.
        points = [key] + [t for t in (round(start, 2), round(end, 2)) if abs(t - key) >= 0.5]
        frames = []
        for j, t in enumerate(points[:3], start=1):
            name = f"e{i}_{j}.jpg"
            if media.extract_frame(src, t, job_dir / name):
                frames.append({"timestamp_sec": t, "file": name})
        rule = rules.lookup(ev["type"])
        description = ev.get("description", "").strip()
        if rule["requirement"]:
            description = f"{description} Требование: {rule['requirement']} ({rule['source']})."
        violations.append({
            "type": rule["title"],
            "type_id": ev["type"],
            "timecode_sec": key,
            "start_sec": start,
            "end_sec": end,
            "confidence": ev.get("confidence"),
            "description": description,
            "evidence_frames": frames,
            "rule": {k: rule[k] for k in ("id", "title", "requirement", "source", "url", "verified")},
        })
    return {
        "verdict": VERDICT_VIOLATION if violations else VERDICT_OK,
        "duration_sec": duration,
        "violations": violations,
    }


def _process(job_id: str, analyze) -> None:
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
    duration = media.probe_duration(src)
    raw = analyze(str(src), deadline)
    result = _build_result(job_id, raw, duration)
    result["timings"] = {"queue_sec": round(t0 - job["created_at"], 2),
                         "processing_sec": round(time.time() - t0, 2)}
    db.finish(job_id, result)
    log.info("job %s done: %s, %d violations, %.1fs",
             job_id, result["verdict"], len(result["violations"]), time.time() - t0)


def _loop(stop: threading.Event) -> None:
    analyze = get_analyzer()
    while not stop.is_set():
        job_id = db.claim_next()
        if job_id is None:
            stop.wait(0.5)
            continue
        try:
            _process(job_id, analyze)
        except Exception as exc:  # любая ошибка анализатора -> failed, сервис живёт дальше
            log.error("job %s failed\n%s", job_id, traceback.format_exc())
            db.fail(job_id, f"{type(exc).__name__}: {exc}")


def start() -> threading.Event:
    stop = threading.Event()
    threading.Thread(target=_loop, args=(stop,), name="cashlens-worker", daemon=True).start()
    return stop
