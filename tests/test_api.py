import base64
import hashlib
import os
import subprocess
import time
from pathlib import Path

# conftest sets these before collection, independent of module import order.
_tmp = os.environ["CASHLENS_DATA_DIR"]

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.media import ffmpeg_exe

KEY = {"X-API-Key": "test-key"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        end = time.monotonic() + 5
        while not c.get("/health").json()["analyzer_ready"]:
            assert time.monotonic() < end, c.get("/health").json()
            time.sleep(0.01)
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
    assert "clip_url" not in v
    assert isinstance(v["type"], str) and v["type"]
    assert isinstance(v["timecode_sec"], float)
    assert "Основание:" in v["description"]  # источник дописан адаптером AI-модуля
    assert 1 <= len(v["evidence_frames"]) <= 3
    first = v["evidence_frames"][0]
    assert first["timestamp_sec"] == v["timecode_sec"]
    assert base64.b64decode(first["image_base64"])[:2] == b"\xff\xd8"  # JPEG
    assert job["analyzer"] == "mock"

    web = client.get(f"/web/jobs/{job['job_id']}").json()
    clip_url = web["violations"][0]["clip_url"]
    clip = client.get(clip_url)
    assert clip.status_code == 200 and clip.headers["content-type"] == "video/mp4"
    assert web["input_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    report = client.get(web["result_json_url"])
    assert report.status_code == 200
    assert web["result_sha256"] == hashlib.sha256(report.content).hexdigest()
    assert report.headers["X-Content-SHA256"] == web["result_sha256"]
    assert "result_sha256" not in report.json() and "feedback" not in report.json()
    assert "image_path" not in report.text
    client.post(f"/web/jobs/{job['job_id']}/feedback", json={"event_index": 0, "decision": "rejected"})
    assert client.get(web["result_json_url"]).content == report.content
    assert client.get(f"/web/jobs/{job['job_id']}").json()["result_sha256"] == web["result_sha256"]
    assert web["violations"][0]["source"]["url"].startswith("https://")
    assert client.get(web["violations"][0]["evidence_frames"][0]["url"]).status_code == 200


def test_health_reports_analyzer(client):
    h = client.get("/health").json()
    assert h["analyzer"] == "mock" and h["analyzer_ready"] is True


def test_feedback_and_rules(client, video, monkeypatch):
    def demo_counts():
        rows = client.get("/web/stats").json()["by_rule"]
        assert all(set(row) == {"rule_id", "confirmed", "rejected"} for row in rows)
        return next((row for row in rows if row["rule_id"] == "demo_event"),
                    {"confirmed": 0, "rejected": 0})

    before = demo_counts()
    monkeypatch.setenv("CASHLENS_MOCK_VERDICT", "violation")
    with open(video, "rb") as f:
        jid = client.post("/web/analyze", files={"file": ("sample.mov", f)}).json()["job_id"]
    _wait(client, jid)
    url = f"/web/jobs/{jid}/feedback"
    assert client.post(url, json={"event_index": 0, "decision": "rejected"}).status_code == 200
    assert client.post(url, json={"event_index": 0, "decision": "confirmed"}).status_code == 200
    assert client.get(f"/web/jobs/{jid}").json()["feedback"] == {"0": "confirmed"}
    after = demo_counts()
    assert after["confirmed"] == before["confirmed"] + 1
    assert after["rejected"] == before["rejected"]
    assert client.post(url, json={"event_index": 5, "decision": "confirmed"}).status_code == 404
    assert client.post(url, json={"event_index": 0, "decision": "maybe"}).status_code == 422

    rules = client.get("/web/rules").json()
    assert rules["analyzer"] == "mock" and isinstance(rules["rules"], list)


def test_db_not_served(client):
    assert client.get("/media/../jobs.sqlite3").status_code == 404


def test_failed_job_cannot_expose_unpublished_json_via_static_alias(client):
    from app import db, main
    job_id = "f" * 32
    directory = main.JOBS_DIR / job_id
    directory.mkdir(exist_ok=True)
    (directory / "result.json").write_text('{"verdict":"no_violation"}', encoding="utf-8")
    (directory / "worker-result.json").write_text("{}", encoding="utf-8")
    db.create(job_id, "synthetic.mov")
    db.fail(job_id, "synthetic failure")
    for filename in ("result.json", "/result.json", "result.json/", "RESULT.JSON", "worker-result.json"):
        assert client.get(f"/media/{job_id}/{filename}").status_code == 404
