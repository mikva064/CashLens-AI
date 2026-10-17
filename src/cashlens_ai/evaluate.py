"""Offline evaluation of saved results by unique video-content group.

Only the manifest and prediction JSON are read. Evidence images, source videos,
model weights and the network are not used by this module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

from pydantic import ValidationError

from .contracts import AnalysisResult
from .errors import DatasetValidationError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = {"development", "holdout", "unused"}


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON object key: {key}")
        output[key] = value
    return output


def _read_json(path: Path, description: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    except (OSError, ValueError, UnicodeError) as exc:
        raise DatasetValidationError(f"Cannot read valid {description} JSON: {exc}") from exc


def _groups_from_manifest(manifest: Any) -> dict[str, dict]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise DatasetValidationError("Expected dataset manifest schema_version 1")
    groups = manifest.get("groups")
    if not isinstance(groups, list):
        raise DatasetValidationError("Manifest groups must be an array")
    by_hash = {}
    for group in groups:
        if not isinstance(group, dict):
            raise DatasetValidationError("Manifest group must be an object")
        sha = group.get("sha256")
        if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
            raise DatasetValidationError("Manifest group has an invalid SHA256")
        if sha in by_hash:
            raise DatasetValidationError("Duplicate SHA256 groups in manifest")
        label = group.get("label")
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise DatasetValidationError("Manifest group label must be 0, 1, or null")
        if group.get("split") not in _SPLITS:
            raise DatasetValidationError("Manifest group has an unknown split")
        filenames = group.get("filenames")
        if not isinstance(filenames, list) or not all(isinstance(name, str) for name in filenames):
            raise DatasetValidationError("Manifest group filenames must be an array of strings")
        points = group.get("weak_points_sec")
        if not isinstance(points, list) or any(
            isinstance(point, bool) or not isinstance(point, (int, float))
            or not math.isfinite(point) or point < 0 for point in points
        ):
            raise DatasetValidationError("Manifest weak points must be finite nonnegative seconds")
        by_hash[sha] = group
    return by_hash


def _validated_result(prediction: dict) -> tuple[AnalysisResult | None, str | None]:
    status = prediction.get("status")
    if status == "failed":
        if set(prediction) != {"sha256", "status", "error"} or not isinstance(prediction["error"], str):
            return None, "Malformed failed prediction entry"
        return None, prediction["error"] or "Prediction job failed"
    if status != "done":
        return None, "Prediction status must be done or failed"
    if set(prediction) != {"sha256", "status", "result"}:
        return None, "Malformed completed prediction entry"
    try:
        # JSON-mode strict validation accepts the contract's serialized Path
        # strings while rejecting coerced string timestamps and nonfinite data.
        serialized = json.dumps(prediction["result"], allow_nan=False)
        result = AnalysisResult.model_validate_json(serialized, strict=True)
        if result.verdict == "requires_review":
            return None, "requires_review: visual observations are not a qualified binary prediction"
        return result, None
    except (TypeError, ValueError, ValidationError):
        return None, "Completed prediction failed AnalysisResult validation"


def evaluate_predictions(
    manifest_path: Path, predictions_path: Path, *, split: str = "holdout"
) -> dict:
    """Compare saved binary verdicts to labels, once per SHA256 video group.

    Missing jobs and failed/invalid completed jobs reduce accuracy_all. Other
    metrics are explicitly conditional on successful processing. Weak point
    offsets are descriptive distances, never an event localization score.
    """
    if split not in _SPLITS:
        raise DatasetValidationError("split must be development, holdout, or unused")
    groups = _groups_from_manifest(_read_json(manifest_path, "manifest"))
    predictions = _read_json(predictions_path, "predictions")
    if not isinstance(predictions, list):
        raise DatasetValidationError("Predictions JSON must be an array")
    by_hash = {}
    ignored_other_split = 0
    ignored_unlabeled = 0
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise DatasetValidationError("Every prediction must be an object with a SHA256")
        sha = prediction.get("sha256")
        if not isinstance(sha, str) or sha not in groups:
            raise DatasetValidationError("Prediction SHA256 is absent from the manifest")
        if sha in by_hash:
            raise DatasetValidationError("Duplicate prediction SHA256; refusing ambiguous results")
        by_hash[sha] = prediction
        if groups[sha]["split"] != split:
            ignored_other_split += 1
        elif groups[sha]["label"] is None:
            ignored_unlabeled += 1

    selected = [group for group in groups.values()
                if group["split"] == split and group["label"] is not None]
    counts = {
        "selected_groups": len(selected),
        "selected_positive_groups": sum(group["label"] == 1 for group in selected),
        "selected_negative_groups": sum(group["label"] == 0 for group in selected),
        "processed": 0, "missing": 0, "failed": 0,
        "excluded_unlabeled_groups": sum(group["split"] == split and group["label"] is None
                                         for group in groups.values()),
        "ignored_other_split_predictions": ignored_other_split,
        "ignored_unlabeled_predictions": ignored_unlabeled,
    }
    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    details = []
    group_offsets = []
    positive_violation_groups = 0
    for group in sorted(selected, key=lambda item: item["sha256"]):
        sha, truth = group["sha256"], group["label"]
        detail = {"sha256": sha, "filenames": group["filenames"], "label": truth,
                  "status": "missing", "predicted_label": None, "correct": False}
        prediction = by_hash.get(sha)
        if prediction is None:
            counts["missing"] += 1
            details.append(detail)
            continue
        result, error = _validated_result(prediction)
        if result is None:
            counts["failed"] += 1
            detail.update(status="failed", error=error)
            details.append(detail)
            continue
        counts["processed"] += 1
        predicted = int(result.verdict == "violation")
        key = ("tp" if predicted else "fn") if truth else ("fp" if predicted else "tn")
        confusion[key] += 1
        detail.update(status="processed", predicted_label=predicted, correct=(truth == predicted))
        if truth == 1 and predicted == 1:
            positive_violation_groups += 1
            points = sorted(set(group["weak_points_sec"]))
            offsets = []
            for violation in result.violations:
                if not points:
                    break
                point = min(points, key=lambda value: (abs(value - violation.timecode_sec), value))
                offsets.append({"predicted_timecode_sec": violation.timecode_sec,
                                "nearest_weak_point_sec": point,
                                "abs_offset_sec": abs(point - violation.timecode_sec)})
            detail["weak_point_offsets"] = offsets
            detail["weak_point_min_abs_offset_sec"] = min(
                (offset["abs_offset_sec"] for offset in offsets), default=None
            )
            if offsets:
                group_offsets.append(detail["weak_point_min_abs_offset_sec"])
        details.append(detail)

    correct = confusion["tp"] + confusion["tn"]
    denominators = {
        "accuracy_processed": counts["processed"],
        "accuracy_all": counts["selected_groups"],
        "precision_processed": confusion["tp"] + confusion["fp"],
        "recall_processed": confusion["tp"] + confusion["fn"],
    }
    numerators = {"accuracy_processed": correct, "accuracy_all": correct,
                  "precision_processed": confusion["tp"], "recall_processed": confusion["tp"]}
    metrics = {name: numerators[name] / denominator if denominator else None
               for name, denominator in denominators.items()}
    metrics.update(numerators=numerators, denominators=denominators)
    return {
        "schema_version": 1,
        "split": split,
        "unit": "unique_sha256_video_group",
        "counts": counts,
        "confusion": confusion,
        "metrics": metrics,
        "weak_point_offsets": {
            "descriptive_only": True,
            "positive_completed_violation_groups": positive_violation_groups,
            "groups_with_weak_points": len(group_offsets),
            "mean_group_nearest_abs_offset_sec": (
                sum(group_offsets) / len(group_offsets) if group_offsets else None
            ),
            "interpretation": "Each group value is the smallest absolute distance between any "
                              "predicted event time and any weak annotation point. No tolerance, "
                              "event matching, event coverage, or localization score is inferred.",
        },
        "limitations": [
            "no_violation means no event was detected under the active rule registry; it is not "
            "a statement of general legal compliance.",
            "Confusion, precision, recall and accuracy_processed exclude missing and failed jobs. "
            "accuracy_all counts every selected missing or failed group as incorrect.",
            "Weak timestamps are annotation points, not event intervals. Types and localization "
            "are not scored because reviewed type and interval ground truth is absent.",
            "per_group contains private filenames, hashes, errors and timestamp details; keep "
            "the report local.",
        ],
        "per_group": details,
    }
