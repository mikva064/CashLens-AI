"""Local, SHA-deduplicated evaluation: clean holdout or exposed reference regression.

Only compact IDs/counts/timestamps are saved. Captions, evidence, filenames and
reference labels are never sent to the model; multipart filenames are fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.parse import urlparse

import httpx
from dotenv import dotenv_values

LIMIT_SEC = 300
SHA = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
REFERENCE_CLASS_MAP = {
    "cash_withheld_then_remainder_returned": "cash_split_remainder_returned",
    "cash_accepted_without_visible_id_check": "cash_receipt_id_not_visible",
    "cash_transferred_to_other_person": "received_cash_handed_other_person",
}


class BenchmarkError(ValueError):
    """Fixed, public-safe evaluation failure reason."""


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BenchmarkError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sha_set(value):
    if not isinstance(value, list) or any(not isinstance(sha, str) or not SHA.fullmatch(sha) for sha in value):
        raise BenchmarkError("Invalid protocol SHA list")
    return set(value)


def reference_events(entry):
    output = []
    for event in entry.get("events", []):
        original = event.get("reference_class_id") or event.get("rule_id")
        if not isinstance(original, str) or not IDENTIFIER.fullmatch(original):
            continue
        points = event.get("evidence_sec", [])
        if not isinstance(points, list):
            raise BenchmarkError("Invalid reference timestamps")
        points = [float(point) for point in points if type(point) in (int, float) and math.isfinite(point) and point >= 0]
        output.append({"reference_class_id": original, "class_id": REFERENCE_CLASS_MAP.get(original, original),
                       "weak_points_sec": sorted(set(points))})
    return output


def load_selection(manifest_path, protocol_path, index_path, suite):
    manifest, protocol, index = map(read_json, (manifest_path, protocol_path, index_path))
    if any(document.get("schema_version") != 1 for document in (manifest, protocol, index)):
        raise BenchmarkError("Unsupported evaluation metadata schema")
    groups = {}
    for entry in manifest["groups"]:
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not SHA.fullmatch(sha) or sha in groups:
            raise BenchmarkError("Manifest SHA groups must be valid and unique")
        groups[sha] = entry
    references = {}
    for entry in index["videos"]:
        sha = entry.get("sha256")
        if not isinstance(sha, str) or sha not in groups:
            raise BenchmarkError("Reference SHA is absent from the canonical manifest")
        label = entry.get("binary_label")
        uncertain = entry.get("expected_verdict") == "uncertain" or label is None
        if not uncertain and (type(label) is not int or label not in (0, 1)
                              or entry.get("expected_verdict") != ("violation" if label else "no_violation")):
            raise BenchmarkError("Inconsistent canonical reference label")
        if sha in references:
            previous = references[sha]
            if previous["binary_label"] != label or previous["uncertain"] != uncertain:
                raise BenchmarkError("Conflicting duplicate reference SHA")
            previous["reference_events"].extend(reference_events(entry))
            continue
        references[sha] = {"binary_label": label, "uncertain": uncertain,
                           "reference_events": reference_events(entry)}
    uncertain = sha_set(protocol.get("uncertain_reference_sha256", [])) | {
        sha for sha, entry in references.items() if entry["uncertain"]}
    exposed = sha_set(protocol.get("reference_sha256", [])) | sha_set(protocol.get("exposed_original_holdout_sha256", [])) | set(references)
    if suite == "clean_holdout":
        requested = sha_set(protocol["remaining_original_holdout_sha256"])
        selected = requested - exposed - uncertain
    elif suite == "reference_regression":
        selected = set(references) - uncertain
    else:
        raise BenchmarkError("Unknown evaluation suite")
    cases = []
    for sha in sorted(selected):
        if sha not in groups:
            raise BenchmarkError("Selected SHA is absent from manifest")
        group = groups[sha]
        label = references[sha]["binary_label"] if suite == "reference_regression" else group.get("label")
        if type(label) is not int or label not in (0, 1):
            raise BenchmarkError("Selected group lacks a definite binary label")
        if suite == "clean_holdout" and group.get("split") != "holdout":
            raise BenchmarkError("Clean selection contains a non-holdout group")
        names = group.get("filenames")
        if not isinstance(names, list) or not names or any(not isinstance(name, str) or Path(name).name != name
                                                         or "/" in name or "\\" in name for name in names):
            raise BenchmarkError("Invalid source basenames")
        cases.append({"sha256": sha, "label": label, "filenames": names,
                      "reference_events": references[sha]["reference_events"] if suite == "reference_regression" else []})
    if not cases:
        raise BenchmarkError("Evaluation selection is empty")
    return cases, {"suite": suite, "unique_videos": len(cases),
                   "positive": sum(case["label"] == 1 for case in cases),
                   "negative": sum(case["label"] == 0 for case in cases),
                   "uncertain_reference_excluded": len(uncertain),
                   "exposed_original_holdout_excluded": len(sha_set(protocol.get("exposed_original_holdout_sha256", [])))}


def resolve_video(directory, case):
    for name in case["filenames"]:
        source = Path(directory) / name
        if source.is_file() and file_sha256(source) == case["sha256"]:
            return source
    raise BenchmarkError("No source file matches the selected SHA")


def compact_events(web):
    output = []
    for kind, key, identifier in (("qualified", "violations", "rule_id"), ("observation", "observations", "observation_id")):
        events = web.get(key, [])
        if not isinstance(events, list):
            raise BenchmarkError("Invalid completed event list")
        for event in events:
            class_id, stamp = event.get(identifier), event.get("timecode_sec")
            if not isinstance(class_id, str) or not IDENTIFIER.fullmatch(class_id) or type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0:
                raise BenchmarkError("Invalid compact event metadata")
            output.append({"kind": kind, "class_id": class_id, "timecode_sec": stamp})
    return output


def run_one(client: httpx.Client, video: Path, *, limit_sec=LIMIT_SEC, poll_interval=1) -> tuple[dict, float]:
    started = time.monotonic()
    try:
        with Path(video).open("rb") as source:
            response = client.post("/api/v1/analyze", files={"file": ("video.mov", source, "video/quicktime")}, timeout=min(120, limit_sec))
        response.raise_for_status()
        job_id = response.json().get("job_id")
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise BenchmarkError("Invalid job ID")
        while time.monotonic() - started < limit_sec:
            remaining = limit_sec - (time.monotonic() - started)
            response = client.get(f"/web/jobs/{job_id}", timeout=max(0.01, min(10, remaining)))
            response.raise_for_status()
            web = response.json()
            if web.get("status") == "failed":
                code = web.get("failure_code")
                return {"status": "failed", "failure_code": code if isinstance(code, str) and IDENTIFIER.fullmatch(code) else "job_failed"}, time.monotonic() - started
            if web.get("status") == "done":
                events = compact_events(web)
                verdict = web.get("verdict")
                if verdict not in {"violation", "no_violation", "requires_review"}:
                    raise BenchmarkError("Unknown web verdict")
                expected = ("violation" if any(event["kind"] == "qualified" for event in events)
                            else "requires_review" if events else "no_violation")
                if verdict != expected:
                    raise BenchmarkError("Web verdict disagrees with its qualified and observed events")
                response = client.get(f"/api/v1/jobs/{job_id}", timeout=max(0.01, min(10, limit_sec - (time.monotonic() - started))))
                response.raise_for_status()
                api = response.json()
                review = verdict == "requires_review"
                if review and (api.get("status") != "failed" or api.get("failure_code") != "requires_review" or "verdict" in api):
                    raise BenchmarkError("Review abstention must not become a jury binary verdict")
                if not review and (api.get("status") != "done" or api.get("verdict") != verdict):
                    raise BenchmarkError("API and web qualified verdict disagree")
                return {"status": "requires_review" if review else "done", "api_status": api["status"],
                        "api_verdict": api.get("verdict"), "api_abstention": review,
                        "candidate_detected": bool(events), "qualified_count": sum(event["kind"] == "qualified" for event in events),
                        "observation_count": sum(event["kind"] == "observation" for event in events), "events": events}, time.monotonic() - started
            if web.get("status") not in {"processing", "queued"}:
                raise BenchmarkError("Unknown job status")
            time.sleep(min(poll_interval, max(0, limit_sec - (time.monotonic() - started))))
        return {"status": "failed", "failure_code": "client_deadline"}, time.monotonic() - started
    except Exception as error:
        # No paths, captions, server error bodies, or API keys in outputs.
        return {"status": "failed", "failure_code": "request_or_contract_error", "error_type": type(error).__name__}, time.monotonic() - started


def runtime_metadata(health, *, require_ready=True):
    if health.get("analyzer") != "model" or (require_ready and health.get("analyzer_ready") is not True):
        raise BenchmarkError("Wait for the real local model before evaluation")
    revision, catalog = health.get("pipeline_revision"), health.get("catalog_hash")
    if not isinstance(revision, str) or not revision or not isinstance(catalog, str) or not SHA.fullmatch(catalog):
        raise BenchmarkError("Runtime must expose pipeline_revision and catalog_hash")
    return {"pipeline_revision": revision, "catalog_hash": catalog, "model_name": health.get("model_name"),
            "performance": health.get("performance", {})}


def wait_runtime(client, expected=None, timeout_sec=210):
    until = time.monotonic() + timeout_sec
    while True:
        response = client.get("/health"); response.raise_for_status()
        health = response.json()
        current = runtime_metadata(health, require_ready=False)
        if expected is not None and current != expected:
            raise BenchmarkError("Runtime version/settings changed during evaluation")
        if health.get("analyzer_ready") is True:
            return current
        if health.get("analyzer_state") == "failed" or time.monotonic() >= until:
            raise BenchmarkError("Local model did not become ready")
        time.sleep(1)


def run_metadata(cases, selection, paths, runtime):
    return {"suite": selection["suite"], **runtime, "manifest_sha256": file_sha256(paths[0]),
            "protocol_sha256": file_sha256(paths[1]), "reference_index_sha256": file_sha256(paths[2]),
            "selected_sha256": [case["sha256"] for case in cases], "reference_class_map": REFERENCE_CLASS_MAP}


def read_resume(path, metadata):
    cached = read_json(path)
    if not isinstance(cached, dict) or cached.get("schema_version") != 2 or cached.get("run") != metadata:
        raise BenchmarkError("Resume refused: evaluation inputs, pipeline, catalog, or settings changed")
    rows = cached.get("predictions")
    allowed = {"sha256", "status", "failure_code", "error_type", "elapsed_sec", "api_status", "api_verdict",
               "api_abstention", "candidate_detected", "qualified_count", "observation_count", "events"}
    if not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) - allowed for row in rows):
        raise BenchmarkError("Resume predictions contain unsupported or non-compact fields")
    if len({row.get("sha256") for row in rows}) != len(rows) or any(row.get("sha256") not in metadata["selected_sha256"] for row in rows):
        raise BenchmarkError("Invalid resume predictions")
    for row in rows:
        if row.get("status") not in {"done", "requires_review", "failed"}:
            raise BenchmarkError("Invalid resume prediction status")
        elapsed = row.get("elapsed_sec")
        if elapsed is not None and (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
            raise BenchmarkError("Invalid resume elapsed time")
        if row["status"] == "failed":
            continue
        events = row.get("events")
        if not isinstance(events, list) or any(not isinstance(event, dict)
            or set(event) != {"kind", "class_id", "timecode_sec"}
            or event["kind"] not in {"qualified", "observation"}
            or not isinstance(event["class_id"], str) or not IDENTIFIER.fullmatch(event["class_id"])
            or type(event["timecode_sec"]) not in (int, float) or not math.isfinite(event["timecode_sec"])
            or event["timecode_sec"] < 0 for event in events):
            raise BenchmarkError("Invalid compact resume events")
        if type(row.get("candidate_detected")) is not bool or row["candidate_detected"] != bool(events):
            raise BenchmarkError("Invalid resume candidate verdict")
    return rows


def save_predictions(path, metadata, rows):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps({"schema_version": 2, "run": metadata, "predictions": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def summarize(cases, rows, selection, metadata):
    by_sha = {row["sha256"]: row for row in rows}
    confusion, legal = Counter(), Counter()
    offsets, elapsed = [], []
    completed = correct = failed = abstentions = typed = type_hits = 0
    for case in cases:
        row = by_sha.get(case["sha256"], {"status": "missing"})
        expected = {event["class_id"] for event in case["reference_events"]}
        typed += len(expected)
        if type(row.get("elapsed_sec")) in (int, float):
            elapsed.append(row["elapsed_sec"])
        if row.get("status") not in {"done", "requires_review"}:
            failed += 1
            legal["failure_or_missing"] += 1
            continue
        completed += 1
        abstentions += bool(row.get("api_abstention"))
        legal["requires_review" if row.get("api_abstention") else row.get("api_verdict", "unknown")] += 1
        detected, truth = bool(row["candidate_detected"]), bool(case["label"])
        confusion[("tp" if detected else "fn") if truth else ("fp" if detected else "tn")] += 1
        correct += detected == truth
        for class_id in expected:
            times = [event["timecode_sec"] for event in row.get("events", []) if event["class_id"] == class_id]
            type_hits += bool(times)
            points = [point for event in case["reference_events"] if event["class_id"] == class_id for point in event["weak_points_sec"]]
            if times and points:
                offsets.append(min(abs(stamp - point) for stamp in times for point in points))
    ratio = lambda value, total: value / total if total else None
    return {"schema_version": 2, "selection": selection, "run": metadata,
            "counts": {"selected": len(cases), "completed_visual_analysis": completed, "failures_or_missing": failed,
                       "qualified_api_abstentions": abstentions},
            "candidate_detection": {"definition": "At least one visual observation or qualified event; compared to weak/general binary labels, not legal ground truth",
                "confusion_completed": {key: confusion[key] for key in ("tp", "fp", "tn", "fn")},
                "accuracy_all": ratio(correct, len(cases)), "accuracy_completed": ratio(correct, completed),
                "precision_completed": ratio(confusion["tp"], confusion["tp"] + confusion["fp"]),
                "recall_completed": ratio(confusion["tp"], confusion["tp"] + confusion["fn"])},
            "qualified_api_outcomes": dict(legal),
            "typed_reference_regression": {"expected_classes_per_video": typed, "matched_classes_per_video": type_hits,
                "type_hit_fraction": ratio(type_hits, typed), "weak_point_matches": len(offsets),
                "mean_nearest_weak_point_offset_sec": statistics.mean(offsets) if offsets else None,
                "localization_score": None},
            "timings": {"recorded": len(elapsed), "mean_sec": statistics.mean(elapsed) if elapsed else None,
                        "max_sec": max(elapsed, default=None), "within_300sec": sum(value <= LIMIT_SEC for value in elapsed)},
            "limitations": ["Reference regression contains exposed user examples, not an independent quality estimate.",
                "Uncertain references are excluded. Unspecified positive labels have no type/time denominator.",
                "Candidate detections are separate from qualified legal/API verdicts; no legal accuracy is scored.",
                "All-video candidate accuracy counts failures/missing as incorrect; conditional metrics exclude them.",
                "A nearest weak point is descriptive, not event matching, interval IoU, or an official localization score."]}


def markdown_report(report):
    ratio = lambda value: "N/A" if value is None else f"{100 * value:.1f}%"
    counts, candidate, types, timing = (report[key] for key in ("counts", "candidate_detection", "typed_reference_regression", "timings"))
    return "\n".join([f"# CashLens evaluation: {report['selection']['suite']}", "",
        f"Pipeline: {report['run']['pipeline_revision']}; catalog: {report['run']['catalog_hash']}",
        f"Unique videos: {counts['selected']}; completed visual analysis: {counts['completed_visual_analysis']}; failures/missing: {counts['failures_or_missing']}; qualified API abstentions: {counts['qualified_api_abstentions']}.", "",
        "| Visual candidate metric (not legal accuracy) | Value |", "|---|---|",
        f"| Accuracy, all selected videos | {ratio(candidate['accuracy_all'])} |",
        f"| Accuracy, completed visual analysis | {ratio(candidate['accuracy_completed'])} |",
        f"| Precision / recall, completed | {ratio(candidate['precision_completed'])} / {ratio(candidate['recall_completed'])} |",
        f"| Reference class hits / expected classes | {types['matched_classes_per_video']} / {types['expected_classes_per_video']} |",
        f"| Responses within 300s | {timing['within_300sec']} / {counts['selected']} |", "",
        "Qualified API outcomes: " + json.dumps(report["qualified_api_outcomes"], sort_keys=True), "",
        *["- " + limitation for limitation in report["limitations"]], ""])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("local/audit/manifest.json"))
    parser.add_argument("--protocol", type=Path, default=Path("local/annotations/reference-protocol.json"))
    parser.add_argument("--reference-index", type=Path, default=Path("local/annotations/reference-index.json"))
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--suite", choices=["clean_holdout", "reference_regression"])
    parser.add_argument("--split", choices=["holdout", "development"], help="Legacy alias: holdout=clean_holdout, development=reference_regression")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", help="Legacy override; prefer CASHLENS_API_KEY or local .env")
    parser.add_argument("--output", type=Path, default=Path("local/bench"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Selection counts only; no server or video access")
    args = parser.parse_args(argv)
    try:
        suite = args.suite or ("reference_regression" if args.split == "development" else "clean_holdout")
        if args.split and args.suite and args.suite != ("clean_holdout" if args.split == "holdout" else "reference_regression"):
            raise BenchmarkError("Conflicting suite and legacy split")
        paths = (args.manifest, args.protocol, args.reference_index)
        cases, selection = load_selection(*paths, suite)
        if args.dry_run:
            print(json.dumps(selection, sort_keys=True))
            return 0
        parsed = urlparse(args.url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password:
            raise BenchmarkError("Evaluation is restricted to the local computer")
        key = args.api_key or os.getenv("CASHLENS_API_KEY") or dotenv_values(".env").get("CASHLENS_API_KEY")
        if not key:
            raise BenchmarkError("Set CASHLENS_API_KEY before evaluation")
        args.output.mkdir(parents=True, exist_ok=True)
        predictions_path = args.output / "predictions.json"
        if predictions_path.exists() and not args.resume:
            raise BenchmarkError("Output already exists; choose a new output directory or use --resume")
        with httpx.Client(base_url=args.url.rstrip("/"), headers={"X-API-Key": key}) as client:
            runtime = wait_runtime(client)
            metadata = run_metadata(cases, selection, paths, runtime)
            rows = read_resume(predictions_path, metadata) if args.resume and predictions_path.exists() else []
            completed_shas = {row["sha256"] for row in rows}
            for index, case in enumerate(cases, 1):
                if case["sha256"] in completed_shas:
                    continue
                wait_runtime(client, runtime)
                try:
                    outcome, elapsed = run_one(client, resolve_video(args.videos, case))
                except Exception as error:
                    outcome, elapsed = {"status": "failed", "failure_code": "source_hash_mismatch", "error_type": type(error).__name__}, None
                response = client.get("/health"); response.raise_for_status()
                if runtime_metadata(response.json(), require_ready=False) != runtime:
                    raise BenchmarkError("Runtime version/settings changed during evaluation")
                measured = round(elapsed, 3) if elapsed is not None else None
                rows.append({"sha256": case["sha256"], **outcome, "elapsed_sec": measured})
                save_predictions(predictions_path, metadata, rows)
                print(json.dumps({"video": index, "total": len(cases), "status": outcome["status"],
                                  "qualified": outcome.get("qualified_count", 0), "observations": outcome.get("observation_count", 0), "elapsed_sec": measured}))
        report = summarize(cases, rows, selection, metadata)
        (args.output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / "benchmark.md").write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps({"suite": suite, **report["counts"]}, sort_keys=True))
        return 0
    except Exception as error:
        # Deliberately do not stringify exceptions: request errors can contain URLs/keys or model payloads.
        reason = str(error) if isinstance(error, BenchmarkError) else type(error).__name__
        print("Evaluation stopped: " + reason, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
