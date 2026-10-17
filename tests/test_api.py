import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ["CASHLENS_DATA_DIR"] = _tmp
os.environ["CASHLENS_API_KEY"] = "test-key"
os.environ["CASHLENS_MOCK_DELAY"] = "0"

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.media import ffmpeg_exe

KEY = {"X-API-Key": "test-key"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def video() -> Path:
    path = Path(_tmp) / "sample.mov"
    subprocess.run(
        [ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "testsrc=size=464x848:rate=10",
         "-t", "6", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def _wait(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/v1/jobs/{job_id}", headers=KEY).json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.3)
    raise AssertionError("задача не завершилась")


def test_requires_api_key(client, video):
    with open(video, "rb") as f:
        r = client.post("/api/v1/analyze", files={"file": ("a.mov", f)})
    assert r.status_code == 401
    assert client.get("/api/v1/jobs/x", headers={"X-API-Key": "wrong"}).status_code == 401


def test_rejects_non_video(client):
    r = client.post("/api/v1/analyze", headers=KEY, files={"file": ("a.txt", b"hi")})
    assert r.status_code == 415
    r = client.post("/api/v1/analyze", headers=KEY, files={"file": ("a.mov", b"not a video")})
    assert r.status_code == 422


def test_unknown_job(client):
    assert client.get("/api/v1/jobs/nope", headers=KEY).status_code == 404


def test_full_flow(client, video):
    with open(video, "rb") as f:
        r = client.post("/api/v1/analyze", headers=KEY, files={"file": ("sample.mov", f)})
    assert r.status_code == 200
    assert r.json()["status"] == "processing"
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert job["verdict"] in ("no_violation", "violation")
    assert (job["verdict"] == "violation") == bool(job["violations"])


def test_result_schema(client, video, monkeypatch):
    """Схема ответа из ТЗ, раздел 10: violations[].type/timecode_sec/description/evidence_frames."""
    monkeypatch.setenv("CASHLENS_MOCK_VERDICT", "violation")
    with open(video, "rb") as f:
        r = client.post("/api/v1/analyze", headers=KEY, files={"file": ("sample.mov", f)})
    job = _wait(client, r.json()["job_id"])
    assert {"job_id", "status", "verdict", "violations"} <= job.keys()
    assert job["verdict"] == "violation"
    v = job["violations"][0]
    assert isinstance(v["type"], str) and v["type"]
    assert isinstance(v["timecode_sec"], float)
    assert "Основание:" in v["description"]  # источник дописан адаптером AI-модуля
    assert 1 <= len(v["evidence_frames"]) <= 3
    first = v["evidence_frames"][0]
    assert first["timestamp_sec"] == v["timecode_sec"]
    assert base64.b64decode(first["image_base64"])[:2] == b"\xff\xd8"  # JPEG
    assert job["analyzer"] == "mock"

    web = client.get(f"/web/jobs/{job['job_id']}").json()
    assert web["violations"][0]["source"]["url"].startswith("https://")
    assert client.get(web["violations"][0]["evidence_frames"][0]["url"]).status_code == 200


def test_health_reports_analyzer(client):
    h = client.get("/health").json()
    assert h["analyzer"] == "mock" and h["analyzer_ready"] is True


def test_db_not_served(client):
    assert client.get("/media/../jobs.sqlite3").status_code == 404
