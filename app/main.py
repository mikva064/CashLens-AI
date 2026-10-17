import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import db, media, worker
from app.config import (ALLOWED_EXT, API_KEY, JOBS_DIR, JOB_TIMEOUT_SEC,
                        MAX_UPLOAD_MB, WEB_DIR, WEB_UPLOAD_ENABLED)

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
    return {"job_id": job_id, "status": "queued"}


def _job_view(job: dict) -> dict:
    view = {
        "job_id": job["id"],
        "status": job["status"],
        "filename": job["filename"],
        "created_at": job["created_at"],
    }
    if job["status"] == "queued":
        view["queue_position"] = db.queue_position(job["id"])
    if job["status"] == "processing" and time.time() - job["created_at"] > JOB_TIMEOUT_SEC + 30:
        view["status"] = "failed"
        view["error"] = "Обработка не уложилась в лимит времени"
    if job["status"] == "done":
        view.update(job["result"])
        view["processing_time_sec"] = round(job["finished_at"] - job["created_at"], 2)
    if job["status"] == "failed":
        view["error"] = job["error"]
    return view


# ---------- Публичный API (по заданию) ----------

@app.post("/api/v1/analyze", status_code=202, dependencies=[Depends(require_key)])
async def analyze(file: UploadFile = File(...)):
    return await _accept_upload(file)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def get_job(job_id: str):
    job = db.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return _job_view(job)


@app.get("/health")
def health():
    return {"status": "ok", "ffmpeg": bool(media.ffmpeg_exe())}


# ---------- Веб-интерфейс (без ключа в браузере) ----------

@app.post("/web/analyze", status_code=202)
async def web_analyze(file: UploadFile = File(...)):
    if not WEB_UPLOAD_ENABLED:
        raise HTTPException(status_code=403, detail="Загрузка через веб отключена")
    return await _accept_upload(file)


@app.get("/web/jobs/{job_id}")
def web_job(job_id: str):
    return get_job(job_id)


app.mount("/media", StaticFiles(directory=JOBS_DIR), name="media")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")
