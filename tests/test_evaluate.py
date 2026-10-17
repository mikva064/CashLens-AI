"""Evaluation never needs private footage, evidence files, or a model runtime."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cashlens_ai.errors import DatasetValidationError
from cashlens_ai.evaluate import evaluate_predictions


def group(letter, label, *, split="holdout", points=None, filenames=None):
    return {"sha256": letter * 64, "label": label, "split": split,
            "filenames": filenames or [f"{letter}.mov"], "source_rows": [],
            "weak_points_sec": points or [], "review_flags": []}


def result(*times):
    return {
        "schema_version": "1.0", "verdict": "violation" if times else "no_violation",
        "violations": [{
            "type": "Synthetic test event", "timecode_sec": timestamp,
            "description": "Synthetic observation", "rule_id": "synthetic_event",
            "source": {"document": "Synthetic test policy", "section": "Test 1",
                       "url": "https://example.invalid/synthetic-policy"},
            "evidence_frames": [{"timestamp_sec": timestamp,
                                 "image_path": "Z:/not-present/evidence.jpg"}],
        } for timestamp in times],
    }


def done(letter, *times):
    return {"sha256": letter * 64, "status": "done", "result": result(*times)}


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest_path = self.root / "manifest.json"
        self.predictions_path = self.root / "predictions.json"
        self.groups = [
            group("a", 1, points=[10, 20], filenames=["positive.mov", "identical-positive.mov"]),
            group("b", 1, points=[5]),
            group("c", 0),
            group("d", 0),
            group("e", 1),
            group("f", 0),
            group("1", 0, split="development"),
            group("2", None),
        ]
        self.predictions = [done("a", 13), done("b"), done("c", 7), done("d"),
                            {"sha256": "e" * 64, "status": "failed", "error": "timeout"},
                            done("1"), done("2")]

    def write(self, *, groups=None, predictions=None):
        # Match the audit manifest shape. Referenced source videos and evidence
        # deliberately do not exist, so a media read would fail these checks.
        manifest = {"schema_version": 1,
                    "source": {"videos_dir": "Z:/not-present/videos", "labels_path": "Z:/labels.xlsx",
                               "sheet": "labels"},
                    "annotations": [], "videos": [], "summary": {},
                    "groups": self.groups if groups is None else groups}
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.predictions_path.write_text(
            json.dumps(self.predictions if predictions is None else predictions), encoding="utf-8"
        )

    def evaluate(self, **kwargs):
        return evaluate_predictions(self.manifest_path, self.predictions_path, **kwargs)

    def test_content_groups_confusion_and_explicit_denominators(self):
        self.write()
        report = self.evaluate()
        counts = report["counts"]
        self.assertEqual(counts["selected_groups"], 6)
        self.assertEqual((counts["processed"], counts["missing"], counts["failed"]), (4, 1, 1))
        self.assertEqual(counts["excluded_unlabeled_groups"], 1)
        self.assertEqual(counts["ignored_other_split_predictions"], 1)
        self.assertEqual(counts["ignored_unlabeled_predictions"], 1)
        self.assertEqual(report["confusion"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        metrics = report["metrics"]
        self.assertEqual(metrics["accuracy_processed"], 0.5)
        self.assertAlmostEqual(metrics["accuracy_all"], 1 / 3)
        self.assertEqual(metrics["precision_processed"], 0.5)
        self.assertEqual(metrics["recall_processed"], 0.5)
        self.assertEqual(metrics["denominators"], {"accuracy_processed": 4, "accuracy_all": 6,
                                                  "precision_processed": 2, "recall_processed": 2})
        self.assertEqual(metrics["numerators"]["accuracy_all"], 2)
        self.assertEqual(len(report["per_group"]), 6)
        self.assertEqual(report["per_group"][0]["filenames"], ["positive.mov", "identical-positive.mov"])
        self.assertTrue(any("registry" in limitation for limitation in report["limitations"]))

    def test_invalid_done_results_count_failed_and_never_clean(self):
        invalid = []
        conflicting = result(13)
        conflicting["verdict"] = "no_violation"
        invalid.append(conflicting)
        missing_evidence = result(13)
        missing_evidence["violations"][0]["evidence_frames"] = []
        invalid.append(missing_evidence)
        extra_field = result()
        extra_field["extra"] = "unexpected"
        invalid.append(extra_field)
        invalid.append(result("13"))
        invalid.append(result(float("nan")))
        invalid.append({"verdict": "no_violation"})
        for raw in invalid:
            with self.subTest(result=raw):
                self.write(groups=[self.groups[0]], predictions=[
                    {"sha256": "a" * 64, "status": "done", "result": raw}
                ])
                report = self.evaluate()
                self.assertEqual(report["counts"]["failed"], 1)
                self.assertEqual(report["counts"]["processed"], 0)
                self.assertEqual(report["metrics"]["accuracy_all"], 0.0)
                self.assertIsNone(report["metrics"]["accuracy_processed"])
                self.assertEqual(report["per_group"][0]["status"], "failed")

    def test_duplicate_and_unknown_prediction_hashes_fail(self):
        for predictions in ([done("a"), done("a")], [done("1"), done("1")], [done("9")]):
            with self.subTest(predictions=predictions):
                self.write(predictions=predictions)
                with self.assertRaises(DatasetValidationError):
                    self.evaluate()

    def test_missing_jobs_reduce_all_accuracy_and_zero_denominators_are_null(self):
        self.write(predictions=[])
        report = self.evaluate()
        self.assertEqual(report["counts"]["missing"], 6)
        self.assertEqual(report["metrics"]["accuracy_all"], 0.0)
        for metric in ("accuracy_processed", "precision_processed", "recall_processed"):
            self.assertIsNone(report["metrics"][metric])
        empty = self.evaluate(split="unused")
        self.assertEqual(empty["counts"]["selected_groups"], 0)
        self.assertIsNone(empty["metrics"]["accuracy_all"])

    def test_weak_point_distances_are_descriptive_and_only_for_positive_hits(self):
        self.write(predictions=[done("a", 9, 18), done("b"), done("c", 8), done("d")])
        report = self.evaluate()
        offsets = report["weak_point_offsets"]
        self.assertTrue(offsets["descriptive_only"])
        self.assertEqual(offsets["positive_completed_violation_groups"], 1)
        self.assertEqual(offsets["groups_with_weak_points"], 1)
        self.assertEqual(offsets["mean_group_nearest_abs_offset_sec"], 1)
        detail = report["per_group"][0]
        self.assertEqual(detail["weak_point_min_abs_offset_sec"], 1)
        self.assertEqual([item["abs_offset_sec"] for item in detail["weak_point_offsets"]], [1, 2])
        self.assertEqual([item["nearest_weak_point_sec"] for item in detail["weak_point_offsets"]], [10, 20])
        self.assertTrue(all("weak_point_offsets" not in item for item in report["per_group"][1:]))
        self.assertIn("No tolerance", offsets["interpretation"])

    def test_positive_without_points_has_no_invented_localization_measure(self):
        self.write(groups=[group("a", 1)], predictions=[done("a", 8)])
        report = self.evaluate()
        self.assertEqual(report["weak_point_offsets"]["positive_completed_violation_groups"], 1)
        self.assertEqual(report["weak_point_offsets"]["groups_with_weak_points"], 0)
        self.assertIsNone(report["weak_point_offsets"]["mean_group_nearest_abs_offset_sec"])
        self.assertIsNone(report["per_group"][0]["weak_point_min_abs_offset_sec"])

    def test_malformed_envelope_is_failed_not_clean(self):
        malformed = [
            {"sha256": "a" * 64, "status": "unknown"},
            {"sha256": "a" * 64, "status": "done"},
            {"sha256": "a" * 64, "status": "failed", "error": None},
        ]
        for entry in malformed:
            with self.subTest(entry=entry):
                self.write(groups=[self.groups[0]], predictions=[entry])
                report = self.evaluate()
                self.assertEqual(report["counts"]["failed"], 1)
                self.assertEqual(report["counts"]["processed"], 0)

    def test_bad_manifest_and_bad_json_shapes_fail_explicitly(self):
        duplicate = [self.groups[0], copy.deepcopy(self.groups[0])]
        bad_label = group("a", True)
        bad_point = group("a", 1, points=[float("nan")])
        for groups in (duplicate, [bad_label], [bad_point]):
            with self.subTest(groups=groups):
                self.write(groups=groups, predictions=[])
                with self.assertRaises(DatasetValidationError):
                    self.evaluate()
        for predictions in ({"predictions": []}, [None], [{"status": "done"}]):
            with self.subTest(predictions=predictions):
                self.write(predictions=predictions)
                with self.assertRaises(DatasetValidationError):
                    self.evaluate()
        self.write(predictions=[])
        self.predictions_path.write_text('{"x":1,"x":2}', encoding="utf-8")
        with self.assertRaises(DatasetValidationError):
            self.evaluate()
        with self.assertRaises(DatasetValidationError):
            self.evaluate(split="typo")

    def test_evaluation_import_does_not_load_model_or_video_libraries(self):
        source = str(Path(__file__).resolve().parents[1] / "src")
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import cashlens_ai.evaluate; "
            "assert not any(name in sys.modules for name in "
            "('torch', 'transformers', 'av', 'cashlens_ai.backend', 'cashlens_ai.pipeline'))"
        )
        completed = subprocess.run([sys.executable, "-c", code, source], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
