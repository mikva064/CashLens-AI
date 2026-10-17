"""Review-only findings must never become a negative or a fabricated legal result."""
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from app import db, main
from cashlens_ai.contracts import AnalysisResult, EvidenceFrame, Observation, Violation
from test_service_model import RULE, StubModel, service, upload, wait_job, wait_state


class ReviewModel(StubModel):
    def __init__(self, *, mixed=False):
        super().__init__()
        self.mixed = mixed

    def capabilities(self):
        return {**super().capabilities(), "observation_type_count": 1,
                "observation_types": [{"id": "synthetic_action", "title": "Synthetic action"}],
                "pipeline_revision": "test-review-v1", "catalog_hash": "a" * 64}

    def analyze(self, video, output_dir, deadline_sec):
        output_dir.mkdir()
        frame = output_dir / "synthetic.jpg"
        Image.new("RGB", (32, 32), "gray").save(frame)
        evidence = [EvidenceFrame(timestamp_sec=1.0, image_path=frame)]
        observation = Observation(observation_id="synthetic_action", type="Synthetic action",
                                  timecode_sec=1.0, description="An observable action",
                                  evidence_frames=evidence,
                                  review_reason="Additional context must be reviewed.")
        violations = ([Violation(type=RULE.title, timecode_sec=1.0,
                                 description="Synthetic qualified action", rule_id=RULE.id,
                                 source=RULE.source, evidence_frames=evidence)] if self.mixed else [])
        return AnalysisResult(verdict="violation" if violations else "requires_review",
                              violations=violations, observations=[observation])


@pytest.mark.parametrize("mixed", [False, True])
def test_review_projection_clips_hashes_and_api_abstention(service, monkeypatch, mixed):
    def make_clip(src, timecode, target, *, duration_sec):
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"synthetic mp4")
        return {"start_sec": 0, "end_sec": 2}

    monkeypatch.setattr(main.media, "make_event_clip", make_clip)
    with service(ReviewModel(mixed=mixed)) as (client, jobs):
        health = wait_state(client, "ready")
        assert health["pipeline_revision"] == "test-review-v1"
        assert health["catalog_hash"] == "a" * 64
        assert health["observation_type_count"] == len(health["observation_types"]) == 1
        job_id = upload(client).json()["job_id"]
        web = wait_job(client, job_id)
        assert web["status"] == "done" and web["review_required"] is True
        assert web["verdict"] == ("violation" if mixed else "requires_review")
        finding = web["observations"][0]
        assert finding["clip_url"] == f"/media/{job_id}/clips/observation_001.mp4"
        assert finding["evidence_frames"][0]["url"].endswith("/ai/synthetic.jpg")
        assert not {"source", "rule_id"} & finding.keys()
        assert len(web["violations"]) == int(mixed)
        stored = db.get(job_id)
        assert stored["status"] == "done"
        download = client.get(web["result_json_url"])
        assert download.status_code == 200
        assert hashlib.sha256(download.content).hexdigest() == web["result_sha256"]
        assert download.json()["observations"] == web["observations"]
        headers = {"X-API-Key": main.API_KEY}
        public = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        analysis = client.get(f"/api/v1/jobs/{job_id}/analysis", headers=headers)
        assert "observations" not in public and "clip_url" not in str(public)
        assert str(jobs) not in str(public)
        if mixed:
            assert public["status"] == "done" and public["verdict"] == "violation"
            assert len(public["violations"]) == 1
            assert analysis.status_code == 200
            assert "observations" not in analysis.json()
            assert str(jobs) not in analysis.text
            assert (client.post(f"/web/jobs/{job_id}/feedback", json={
                "event_index": 0, "decision": "confirmed"}).status_code == 200)
        else:
            assert public["status"] == "failed" and public["failure_code"] == "requires_review"
            assert "verdict" not in public
            assert analysis.status_code == 409
        # Observation positions cannot accidentally address normative feedback.
        assert client.post(f"/web/jobs/{job_id}/feedback", json={
            "event_index": int(mixed), "decision": "confirmed"}).status_code == 404


def test_missing_observation_clip_fails_without_a_verdict(service, monkeypatch):
    monkeypatch.setattr(main.media, "make_event_clip", lambda *a, **k: {})
    with service(ReviewModel()) as (client, _):
        wait_state(client, "ready")
        job = wait_job(client, upload(client).json()["job_id"])
        assert job["status"] == "failed" and job["failure_code"] == "invalid_result"
        assert "verdict" not in job


@pytest.mark.parametrize("defect", ["outside_evidence", "missing_evidence", "false_negative_verdict"])
def test_invalid_observation_artifact_is_failed_without_leaking_paths(service, monkeypatch, defect):
    class InvalidReviewModel(ReviewModel):
        def analyze(self, video, output_dir, deadline_sec):
            result = super().analyze(video, output_dir, deadline_sec)
            if defect == "false_negative_verdict":
                # Simulate a faulty worker result bypassing constructor validation.
                return result.model_copy(update={"verdict": "no_violation"})
            frame = output_dir.parent.parent / "outside-synthetic.jpg"
            if defect == "outside_evidence":
                Image.new("RGB", (32, 32), "gray").save(frame)
            finding = result.observations[0].model_copy(update={
                "evidence_frames": [EvidenceFrame(timestamp_sec=1, image_path=frame)]})
            return result.model_copy(update={"observations": [finding]})

    def make_clip(src, timecode, target, *, duration_sec):
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"synthetic mp4")
        return {"start_sec": 0, "end_sec": 2}

    monkeypatch.setattr(main.media, "make_event_clip", make_clip)
    with service(InvalidReviewModel()) as (client, jobs):
        wait_state(client, "ready")
        job_id = upload(client).json()["job_id"]
        web = wait_job(client, job_id)
        assert web["status"] == "failed" and web["failure_code"] == "invalid_result"
        assert "verdict" not in web and "observations" not in web
        public = client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": main.API_KEY}).json()
        assert public["status"] == "failed" and "verdict" not in public
        assert str(jobs) not in str(web) and str(jobs) not in str(public)
        assert not (jobs / job_id / "result.json").exists()
