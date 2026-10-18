"""Narrative provenance is explicit in the web view, using synthetic events only."""
import hashlib
import json

from PIL import Image
import pytest

from app.reporting import web_result, write_result_report
from cashlens_ai.contracts import AnalysisResult
from cashlens_ai.integration import to_api_payload


def synthetic_job(tmp_path, *, status="generated", category="Synthetic category criterion", kind="violation"):
    job_id = "a" * 32
    evidence = tmp_path / job_id / "ai" / "synthetic.jpg"
    evidence.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "gray").save(evidence, format="JPEG")
    event = {"type": "Synthetic category", "timecode_sec": 1.0,
             "description": "Synthetic per-event narrative <img src=x onerror=alert(1)>",
             "evidence_frames": [{"timestamp_sec": 1.0, "image_path": str(evidence)}]}
    if status is not None:
        event.update(commentary_status=status, category_description=category)
    if kind == "violation":
        event.update(rule_id="synthetic_rule", source={"document": "Synthetic source",
                     "section": "Section 1", "url": "https://example.com/rule"})
    else:
        event.update(observation_id="synthetic_observation",
                     review_reason="Synthetic facts require human verification.")
    analysis = {"verdict": "violation" if kind == "violation" else "requires_review",
                "violations": [event] if kind == "violation" else [],
                "observations": [event] if kind != "violation" else []}
    return {"id": job_id, "filename": "synthetic.mov", "created_at": 100.0,
            "result": {"analysis": analysis, "analyzer": "model", "duration_sec": 2.0,
                       "completed_at": 102.0, "timings": {}}}


@pytest.mark.parametrize("kind", ["violation", "observation"])
@pytest.mark.parametrize("status", ["generated", "uncertain", "unavailable", "legacy_template"])
def test_web_projection_preserves_commentary_provenance(tmp_path, kind, status):
    job = synthetic_job(tmp_path, status=status, kind=kind)
    key = "violations" if kind == "violation" else "observations"
    projected = web_result(job, tmp_path)[key][0]
    assert projected["commentary_status"] == status
    assert projected["category_description"] == "Synthetic category criterion"
    assert projected["description"] == job["result"]["analysis"][key][0]["description"]
    assert "image_path" not in projected["evidence_frames"][0]
    if kind == "observation":
        assert projected["review_reason"] == "Synthetic facts require human verification."
        assert not {"rule_id", "source"} & projected.keys()


@pytest.mark.parametrize("kind", ["violation", "observation"])
def test_legacy_result_gets_criterion_label_without_rewriting_artifact(tmp_path, kind):
    job = synthetic_job(tmp_path, status=None, kind=kind)
    key = "violations" if kind == "violation" else "observations"
    event = job["result"]["analysis"][key][0]
    event["description"] = "Synthetic historical criterion"
    archive = tmp_path / job["id"] / "result.json"
    original = (json.dumps({"original": "Synthetic immutable archive"}) + "\n").encode()
    archive.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    projected = web_result(job, tmp_path)[key][0]
    assert projected["commentary_status"] == "legacy_template"
    assert projected["category_description"] == event["description"]
    assert projected["description"] == event["description"]
    assert archive.read_bytes() == original
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest
    assert "commentary_status" not in event and "category_description" not in event


@pytest.mark.parametrize("status", ["generated", "uncertain", "unavailable", "legacy_template"])
def test_missing_category_text_has_transparent_fallback(tmp_path, status):
    job = synthetic_job(tmp_path, status=status, category=None)
    event = job["result"]["analysis"]["violations"][0]
    projected = web_result(job, tmp_path)["violations"][0]
    expected = event["description"] if status == "legacy_template" else event["type"]
    assert projected["category_description"] == expected


def test_new_report_contains_provenance_and_exact_download_hash(tmp_path):
    job = synthetic_job(tmp_path)
    digest = write_result_report(job, tmp_path)
    archive = tmp_path / job["id"] / "result.json"
    content = archive.read_bytes()
    assert hashlib.sha256(content).hexdigest() == digest
    event = json.loads(content)["violations"][0]
    assert event["commentary_status"] == "generated"
    assert event["category_description"] == "Synthetic category criterion"


def test_public_api_shape_does_not_gain_web_commentary_fields(tmp_path):
    job = synthetic_job(tmp_path)
    result = AnalysisResult.model_validate(job["result"]["analysis"])
    payload = to_api_payload(result, job_id=job["id"], evidence_root=tmp_path / job["id"] / "ai")
    assert set(payload["violations"][0]) == {"type", "timecode_sec", "description", "evidence_frames"}
    assert not {"commentary_status", "category_description"} & payload["violations"][0].keys()
