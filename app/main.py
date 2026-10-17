import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import db, media, worker
from app.config import (ALLOWED_EXT, API_KEY, JOBS_DIR, JOB_TIMEOUT_SEC, MAX_UPLOAD_MB,
                        ROOT, WEB_DIR, WEB_UPLOAD_ENABLED)
from cashlens_ai.contracts import AnalysisResult
from cashlens_ai.integration import to_api_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    stop = worker.start()
    yield
    stop.set()


app = FastAPI(title="CashLens AI", version="0.1.0", lifespan=lifespan)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")


async def _accept_upload(file: UploadFile) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail=f"Ожидается видео {sorted(ALLOWED_EXT)}")

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    dst = job_dir / "video.mov"
    limit = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with open(dst, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                out.close()
                dst.unlink()
                job_dir.rmdir()
                raise HTTPException(status_code=413, detail=f"Файл больше {MAX_UPLOAD_MB} МБ")
            out.write(chunk)

    if media.probe_duration(dst) is None:
        dst.unlink()
        job_dir.rmdir()
        raise HTTPException(status_code=422, detail="Файл не распознан как видео")

    db.create(job_id, file.filename or "video.mov")
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
        view = {"job_id": job["id"], "status": "done", "verdict": result.verdict, "violations": [
            {"type": v.type, "timecode_sec": v.timecode_sec, "description": v.description,
             "rule_id": v.rule_id, "source": v.source.model_dump(mode="json"),
             "evidence_frames": [
                 {"timestamp_sec": f.timestamp_sec,
                  "url": "/media/" + f.image_path.resolve().relative_to(JOBS_DIR.resolve()).as_posix()}
                 for f in v.evidence_frames]}
            for v in result.violations]}
        view["feedback"] = db.get_feedback(job["id"])
        view["filename"] = job["filename"]
        view["created_at"] = job["created_at"]
        view["video_url"] = f"/media/{job['id']}/preview.mp4"
        view["original_url"] = f"/media/{job['id']}/video.mov"
    view["analyzer"] = stored["analyzer"]
    view["duration_sec"] = stored["duration_sec"]
    view["processing_time_sec"] = round(job["finished_at"] - job["created_at"], 2)
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
            "stage": status}
    if status == "queued":
        view["queue_position"] = db.queue_position(job["id"])
    if status == "processing" and time.time() - job["created_at"] > JOB_TIMEOUT_SEC + 30:
        view["status"] = view["stage"] = "failed"
        view["error"] = "Обработка не уложилась в лимит времени"
    if status == "failed":
        view["error"] = job["error"]
    return view


def _load(job_id: str) -> dict:
    job = db.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job


# ---------- Публичный API (по заданию) ----------

@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def analyze(file: UploadFile = File(...)):
    return await _accept_upload(file)


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
    return {"status": "ok" if worker.state["ready"] else "degraded",
            "analyzer": worker.state["analyzer"], "analyzer_ready": worker.state["ready"],
            "analyzer_error": worker.state["error"]}


# ---------- Веб-интерфейс (без ключа в браузере) ----------

@app.post("/web/analyze")
async def web_analyze(file: UploadFile = File(...)):
    if not WEB_UPLOAD_ENABLED:
        raise HTTPException(status_code=403, detail="Загрузка через веб отключена")
    return await _accept_upload(file)


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
    try:
        rules = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        rules = []
    return {
        "analyzer": worker.state["analyzer"],
        "ready": worker.state["ready"],
        "rules": [{"id": r.get("id"), "title": r.get("title"),
                   "condition": r.get("observable_condition"),
                   "source": r.get("source")} for r in rules if r.get("verified")],
    }


app.mount("/media", StaticFiles(directory=JOBS_DIR), name="media")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")
