import hashlib
import json
import logging
import os
import math
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import db, media, reporting, worker
from app.feedback_stats import feedback_counts
from cashlens_ai.contracts import AnalysisResult
from cashlens_ai.integration import to_api_payload
from app.config import (ALLOWED_EXT, API_KEY, JOBS_DIR, JOB_TIMEOUT_SEC, MAX_OUTSTANDING_JOBS,
                        MAX_UPLOAD_MB, ROOT, WEB_DIR, WEB_UPLOAD_ENABLED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    stop = worker.start()
    try:
        yield
    finally:
        worker.shutdown(stop)


app = FastAPI(title="CashLens AI", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def record_request_start(request: Request, call_next):
    # Recorded before FastAPI parses the multipart body, so upload time counts.
    request.state.started_at = time.time()
    return await call_next(request)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")


async def _accept_upload(file: UploadFile, *, started_at: float | None = None) -> dict:
    analyzer = worker.snapshot()
    if not analyzer["ready"]:
        await file.close()
        message = ("Локальная модель загружается. Дождитесь готовности и повторите загрузку."
                   if analyzer["status"] == "loading" else
                   analyzer["error"] or "Локальная модель ещё не запущена")
        raise HTTPException(status_code=503, detail=message, headers={"Retry-After": "5"})
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        await file.close()
        raise HTTPException(status_code=415, detail=f"Ожидается видео {sorted(ALLOWED_EXT)}")

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    dst = job_dir / "video.mov"
    limit = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    digest = hashlib.sha256()
    try:
        with open(dst, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"Файл больше {MAX_UPLOAD_MB} МБ")
                out.write(chunk)
                digest.update(chunk)
        duration = media.probe_duration(dst)
        if duration is None or not math.isfinite(duration) or duration <= 0:
            raise HTTPException(status_code=422, detail="Файл не распознан как видео")
        admitted = db.create(job_id, file.filename or "video.mov", created_at=started_at,
                             max_outstanding=MAX_OUTSTANDING_JOBS, timeout_sec=JOB_TIMEOUT_SEC,
                             input_sha256=digest.hexdigest())
    except Exception:
        dst.unlink(missing_ok=True)
        job_dir.rmdir()
        raise
    finally:
        await file.close()
    if admitted:
        # A long upload/probe can outlive a permanent model failure and its
        # queue drain. Recheck after INSERT so no newly queued job is stranded.
        latest = worker.snapshot()
        if latest["status"] in {"failed", "stopped"}:
            db.fail(job_id, latest["error"] or "Локальная модель недоступна.", code="model_unavailable")
            admitted = db.get(job_id)["status"] != "failed"
    if not admitted:
        job = db.get(job_id)
        return {"job_id": job_id, "status": "failed", "error": job["error"],
                "failure_code": job["failure_code"]}
    return {"job_id": job_id, "status": "processing"}


def _done_view(job: dict, for_api: bool) -> dict:
    job_dir = JOBS_DIR / job["id"]
    stored = job["result"]
    result = AnalysisResult.model_validate(stored["analysis"])
    if for_api:
        # Точная схема жюри (ТЗ, раздел 10): адаптер AI-модуля кодирует кадры в base64
        # и дописывает нормативный источник в description.
        view = to_api_payload(result, job_id=job["id"], evidence_root=job_dir / "ai")
        for out, v in zip(view["violations"], result.violations):
            out["rule_id"] = v.rule_id
            out["source"] = v.source.model_dump(mode="json")
    else:
        view = reporting.web_result(job, JOBS_DIR)
        view["feedback"] = db.get_feedback(job["id"])
        if job.get("result_sha256"):
            view["result_sha256"] = job["result_sha256"]
            view["result_json_url"] = f"/media/{job['id']}/result.json"
    view["analyzer"] = stored["analyzer"]
    view["duration_sec"] = stored["duration_sec"]
    view["processing_time_sec"] = round((stored.get("completed_at") or job["finished_at"]) - job["created_at"], 2)
    view["timings"] = stored["timings"]
    return view


def _job_view(job: dict, for_api: bool) -> dict:
    """Статусы по ТЗ — только processing / done / failed, поэтому «в очереди»
    наружу отдаётся как processing, а подробность — в поле stage.
    """
    status = job["status"]
    if status == "done":
        return _done_view(job, for_api)
    view = {"job_id": job["id"], "status": "processing" if status == "queued" else status,
            "stage": status,
            "elapsed_sec": round(time.time() - job["created_at"], 1),
            "timeout_sec": JOB_TIMEOUT_SEC}
    if status == "queued":
        view["queue_position"] = db.queue_position(job["id"])
    if status == "failed":
        view["error"] = job["error"]
        view["failure_code"] = job.get("failure_code")
    return view


def _load(job_id: str) -> dict:
    job = db.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job


# ---------- Публичный API (по заданию) ----------

@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def analyze(request: Request, file: UploadFile = File(...)):
    return await _accept_upload(file, started_at=request.state.started_at)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def get_job(job_id: str):
    return _job_view(_load(job_id), for_api=True)


@app.get("/api/v1/jobs/{job_id}/analysis", dependencies=[Depends(require_key)],
         include_in_schema=False)
def get_analysis(job_id: str):
    """Внутренний AnalysisResult для оценки качества (scripts/benchmark.py)."""
    job = _load(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Задача ещё не завершена")
    analysis = job["result"]["analysis"]
    for v in analysis["violations"]:
        for f in v["evidence_frames"]:
            f["image_path"] = Path(f["image_path"]).name  # пути сервера наружу не отдаём
    return analysis


@app.get("/health")
def health():
    analyzer = worker.snapshot()
    return {"status": "ok" if analyzer["ready"] else "degraded",
            "analyzer": analyzer["analyzer"], "analyzer_ready": analyzer["ready"],
            "analyzer_state": analyzer["status"], "analyzer_error": analyzer["error"],
            "model_name": analyzer["model_name"], "rule_count": analyzer["rule_count"],
            "rules": analyzer["rules"], "limitations": analyzer["limitations"],
            "performance": analyzer["performance"],
            "max_outstanding_jobs": MAX_OUTSTANDING_JOBS,
            "web_upload_enabled": WEB_UPLOAD_ENABLED,
            "max_upload_mb": MAX_UPLOAD_MB, "job_timeout_sec": JOB_TIMEOUT_SEC}


# ---------- Веб-интерфейс (без ключа в браузере) ----------

@app.post("/web/analyze")
async def web_analyze(request: Request, file: UploadFile = File(...)):
    if not WEB_UPLOAD_ENABLED:
        raise HTTPException(status_code=403, detail="Загрузка через веб отключена")
    return await _accept_upload(file, started_at=request.state.started_at)


@app.get("/web/jobs/{job_id}")
def web_job(job_id: str):
    # Интерфейсу — ссылки на кадры вместо base64, чтобы опрос раз в 2 с был лёгким.
    return _job_view(_load(job_id), for_api=False)


class Feedback(BaseModel):
    event_index: int = Field(ge=0)
    decision: Literal["confirmed", "rejected"]
    comment: str | None = Field(default=None, max_length=500)


@app.post("/web/jobs/{job_id}/feedback")
def web_feedback(job_id: str, body: Feedback):
    job = _load(job_id)
    if job["status"] != "done" or body.event_index >= len(job["result"]["analysis"]["violations"]):
        raise HTTPException(status_code=404, detail="Событие не найдено")
    db.set_feedback(job_id, body.event_index, body.decision, body.comment)
    return {"ok": True}


@app.get("/web/rules")
def web_rules():
    """Что именно проверяет сервис — показываем пользователю, чтобы вердикт
    «нарушение не обнаружено» не выглядел как чёрный ящик."""
    path = Path(os.getenv("CASHLENS_RULES", ROOT / "config" / "rules.json"))
    if not path.is_absolute():
        path = ROOT / path
    try:
        rules = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        rules = []
    if not isinstance(rules, list):
        rules = []
    return {
        "analyzer": worker.snapshot()["analyzer"],
        "ready": worker.snapshot()["ready"],
        "rules": [{"id": r.get("id"), "title": r.get("title"),
                   "condition": r.get("observable_condition"),
                   "source": r.get("source")} for r in rules if isinstance(r, dict) and r.get("verified")],
    }


@app.get("/web/stats")
def web_stats():
    return feedback_counts(rule["id"] for rule in web_rules()["rules"])


@app.get("/media/{job_id}/result.json", include_in_schema=False)
def result_report(job_id: str):
    job = _load(job_id)
    path = JOBS_DIR / job_id / "result.json"
    if job["status"] != "done" or not job.get("result_sha256") or not path.is_file():
        raise HTTPException(status_code=404, detail="Итоговый отчёт не готов")
    return FileResponse(path, media_type="application/json", filename=f"cashlens-{job_id}.json",
                        headers={"X-Content-SHA256": job["result_sha256"]})


class MediaFiles(StaticFiles):
    async def get_response(self, path, scope):
        # JSON is served only through the completed-job route above. This also
        # blocks aliases normalized by StaticFiles and private worker artifacts.
        if Path(path.rstrip("/\\")).suffix.lower() not in {".mov", ".mp4", ".jpg", ".jpeg", ".png"}:
            raise HTTPException(status_code=404, detail="Файл не найден")
        return await super().get_response(path, scope)


app.mount("/media", MediaFiles(directory=JOBS_DIR), name="media")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")
