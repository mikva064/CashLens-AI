"""Evaluation isolation, abstentions and versioned resume; synthetic files only."""
import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

_spec = importlib.util.spec_from_file_location("cashlens_benchmark", Path(__file__).parents[1] / "scripts/benchmark.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def metadata_files(tmp_path):
    def group(letter, label, split):
        return {"sha256": letter * 64, "label": label, "split": split,
                "filenames": [f"original-{letter}.mov", f"duplicate-{letter}.mov"]}
    manifest = {"schema_version": 1, "groups": [group("a", 1, "holdout"), group("b", 0, "holdout"),
        group("c", 1, "development"), group("d", None, "development"), group("e", 0, "holdout")]}
    protocol = {"schema_version": 1, "reference_sha256": [letter * 64 for letter in "bcd"],
                "remaining_original_holdout_sha256": [letter * 64 for letter in "aaebb"],
                "exposed_original_holdout_sha256": ["b" * 64], "uncertain_reference_sha256": ["d" * 64]}
    positive = {"sha256": "c" * 64, "binary_label": 1, "expected_verdict": "violation",
                "events": [{"reference_class_id": "cash_withheld_then_remainder_returned", "evidence_sec": [2],
                            "notes": "SYNTHETIC_PRIVATE_DESCRIPTION"}]}
    index = {"schema_version": 1, "videos": [{"sha256": "b" * 64, "binary_label": 0,
             "expected_verdict": "no_violation", "events": []}, positive, copy.deepcopy(positive),
             {"sha256": "d" * 64, "binary_label": None, "expected_verdict": "uncertain", "events": []}]}
    return tuple(write(tmp_path / name, data) for name, data in
                 (("manifest.json", manifest), ("protocol.json", protocol), ("index.json", index)))


def test_clean_holdout_excludes_every_exposed_sha_and_deduplicates(metadata_files):
    cases, selection = bench.load_selection(*metadata_files, "clean_holdout")
    assert {case["sha256"] for case in cases} == {"a" * 64, "e" * 64}
    assert selection["unique_videos"] == 2
    assert selection["uncertain_reference_excluded"] == 1
    assert selection["exposed_original_holdout_excluded"] == 1


def test_reference_regression_uses_canonical_labels_mapping_and_uncertain_exclusion(metadata_files):
    cases, selection = bench.load_selection(*metadata_files, "reference_regression")
    assert selection["suite"] == "reference_regression"
    assert len(cases) == 2
    positive = next(case for case in cases if case["label"] == 1)
    assert positive["reference_events"][0]["class_id"] == "cash_split_remainder_returned"
    assert positive["reference_events"][0]["reference_class_id"] == "cash_withheld_then_remainder_returned"
    assert "SYNTHETIC_PRIVATE_DESCRIPTION" not in json.dumps(cases)


def test_conflicting_duplicate_reference_sha_is_rejected(metadata_files):
    index = bench.read_json(metadata_files[2])
    index["videos"][2].update(binary_label=0, expected_verdict="no_violation")
    write(metadata_files[2], index)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        bench.load_selection(*metadata_files, "reference_regression")


def test_resolve_video_verifies_content_not_filename(tmp_path):
    good = tmp_path / "second.mov"
    good.write_bytes(b"synthetic content")
    (tmp_path / "first.mov").write_bytes(b"wrong bytes")
    case = {"sha256": bench.file_sha256(good), "filenames": ["first.mov", "second.mov"]}
    assert bench.resolve_video(tmp_path, case) == good
    good.write_bytes(b"changed")
    with pytest.raises(ValueError, match="matches the selected SHA"):
        bench.resolve_video(tmp_path, case)


@pytest.mark.parametrize("review", [True, False])
def test_web_candidates_are_separate_from_qualified_api_and_never_export_captions(tmp_path, review):
    source = tmp_path / "private-original-name.mov"
    source.write_bytes(b"synthetic video bytes")
    job_id = "a" * 32
    posted = []
    sentinel = "SYNTHETIC_DESCRIPTION_DO_NOT_EXPORT"

    def transport(request):
        if request.method == "POST":
            body = request.read()
            assert b'filename="video.mov"' in body
            assert b"private-original-name" not in body
            posted.append(body)
            return httpx.Response(200, json={"job_id": job_id})
        if request.url.path.startswith("/web/"):
            return httpx.Response(200, json={"status": "done", "verdict": "requires_review" if review else "violation",
                "violations": [] if review else [{"rule_id": "synthetic_rule", "timecode_sec": 2, "description": sentinel,
                                                  "evidence_frames": [{"url": "/media/private.jpg"}]}],
                "observations": [{"observation_id": "cash_split_remainder_returned", "timecode_sec": 2,
                                  "description": sentinel, "review_reason": sentinel, "evidence_frames": []}]})
        return httpx.Response(200, json={"status": "failed", "failure_code": "requires_review", "error": sentinel}
                              if review else {"status": "done", "verdict": "violation", "evidence_frames": [sentinel]})

    with httpx.Client(base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(transport)) as client:
        outcome, elapsed = bench.run_one(client, source, poll_interval=0)
    assert posted and elapsed >= 0
    assert outcome["candidate_detected"] is True
    assert outcome["api_abstention"] is review
    assert outcome["status"] == ("requires_review" if review else "done")
    assert outcome["qualified_count"] == (0 if review else 1)
    serialized = json.dumps(outcome)
    assert sentinel not in serialized and "evidence_frames" not in serialized and "private.jpg" not in serialized


def test_request_failure_redacts_server_body_and_keys(tmp_path):
    source = tmp_path / "synthetic.mov"
    source.write_bytes(b"synthetic")
    with httpx.Client(base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(
        lambda request: httpx.Response(500, json={"detail": "SYNTHETIC_PRIVATE_SECRET"}))) as client:
        outcome, _ = bench.run_one(client, source)
    assert outcome["status"] == "failed"
    assert "SYNTHETIC_PRIVATE_SECRET" not in json.dumps(outcome)


def test_resume_rejects_changed_pipeline_catalog_settings_or_old_cache(tmp_path):
    metadata = {"pipeline_revision": "r1", "catalog_hash": "a" * 64, "selected_sha256": ["b" * 64], "performance": {"profile": "balanced"}}
    path = tmp_path / "predictions.json"
    bench.save_predictions(path, metadata, [{"sha256": "b" * 64, "status": "failed", "failure_code": "job_failed", "elapsed_sec": 1}])
    assert len(bench.read_resume(path, metadata)) == 1
    for key, value in (("pipeline_revision", "r2"), ("catalog_hash", "c" * 64), ("performance", {"profile": "quality"})):
        with pytest.raises(ValueError, match="Resume refused"):
            bench.read_resume(path, {**metadata, key: value})
    write(path, [])
    with pytest.raises(ValueError, match="Resume refused"):
        bench.read_resume(path, metadata)


def test_resume_never_reexports_captions_or_evidence_from_foreign_cache(tmp_path):
    metadata = {"selected_sha256": ["a" * 64]}
    path = tmp_path / "predictions.json"
    bench.save_predictions(path, metadata, [{"sha256": "a" * 64, "status": "failed", "description": "SYNTHETIC_PRIVATE_TEXT"}])
    with pytest.raises(ValueError, match="non-compact"):
        bench.read_resume(path, metadata)


def test_summary_counts_review_as_candidate_but_abstains_from_legal_and_keeps_failed_type_denominator():
    ref = [{"reference_class_id": "original_class", "class_id": "runtime_class", "weak_points_sec": [2]}]
    cases = [{"sha256": "a" * 64, "label": 1, "reference_events": ref},
             {"sha256": "b" * 64, "label": 0, "reference_events": []},
             {"sha256": "c" * 64, "label": 1, "reference_events": ref},
             {"sha256": "d" * 64, "label": 1, "reference_events": []}]
    rows = [{"sha256": "a" * 64, "status": "requires_review", "api_abstention": True, "candidate_detected": True,
             "elapsed_sec": 12, "events": [{"class_id": "runtime_class", "timecode_sec": 2}]},
            {"sha256": "b" * 64, "status": "done", "api_verdict": "no_violation", "candidate_detected": False,
             "elapsed_sec": 10, "events": []},
            {"sha256": "c" * 64, "status": "failed", "elapsed_sec": 300},
            {"sha256": "d" * 64, "status": "failed", "elapsed_sec": None}]
    report = bench.summarize(cases, rows, {"suite": "reference_regression"}, {})
    assert report["counts"] == {"selected": 4, "completed_visual_analysis": 2, "failures_or_missing": 2, "qualified_api_abstentions": 1}
    assert report["candidate_detection"]["accuracy_all"] == 0.5
    assert report["candidate_detection"]["accuracy_completed"] == 1
    assert report["qualified_api_outcomes"]["requires_review"] == 1
    assert report["typed_reference_regression"]["expected_classes_per_video"] == 2
    assert report["typed_reference_regression"]["matched_classes_per_video"] == 1
    assert report["typed_reference_regression"]["localization_score"] is None
    assert report["timings"]["recorded"] == 3


def test_dry_run_never_connects_to_service_or_reads_video(metadata_files, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench.httpx, "Client", lambda *args, **kwargs: pytest.fail("Dry run must not access service"))
    args = ["--manifest", str(metadata_files[0]), "--protocol", str(metadata_files[1]),
            "--reference-index", str(metadata_files[2]), "--videos", str(tmp_path / "nonexistent"), "--dry-run"]
    assert bench.main(args) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["unique_videos"] == 2
    assert "original-" not in output and "SYNTHETIC_PRIVATE_DESCRIPTION" not in output
