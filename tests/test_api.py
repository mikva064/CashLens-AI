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
    assert r.status_code == 202
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert job["verdict"] in ("no_violation", "possible_violation")
    assert (job["verdict"] == "possible_violation") == bool(job["events"])
    for ev in job["events"]:
        assert ev["rule"]["source"]
        assert ev["evidence_frames"]
        assert client.get(ev["evidence_frames"][0]["url"]).status_code == 200
    assert client.get(job["video_url"]).status_code == 200


def test_db_not_served(client):
    assert client.get("/media/../jobs.sqlite3").status_code == 404
