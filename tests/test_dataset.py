"""Meaningful audit checks use synthetic bytes; no private videos are bundled."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

try:
    from openpyxl import Workbook
except ImportError:
    Workbook = None

from cashlens_ai.dataset import audit_dataset, parse_timestamp_seconds
from cashlens_ai.errors import DatasetValidationError


class TimestampTests(unittest.TestCase):
    def test_weak_points_support_source_notation(self):
        cases = {"11 секунда": 11, "1минута 8 секунда": 68,
                 "1 мин. 8 сек.": 68, "01:08": 68, "1:01:08.5": 3668.5,
                 "0,5": 0.5, 0.5: 0.5, "22": 22}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(parse_timestamp_seconds(value), expected)
        self.assertIsNone(parse_timestamp_seconds(None))
        self.assertIsNone(parse_timestamp_seconds(" "))

    def test_bad_points_fail(self):
        for value in (-1, float("nan"), True, "01:70", "at 11 sec", "1 sec 2 sec"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_timestamp_seconds(value)


@unittest.skipIf(Workbook is None, "openpyxl data extra is not installed")
class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        self.labels = self.root / "labels.xlsx"
        self.output = self.root / "audit"

    def workbook(self, rows):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "labels"
        sheet.append(["наименование файла", "таймстемп", "нарушение / без нарушения"])
        for row in rows:
            sheet.append(row)
        workbook.save(self.labels)
        workbook.close()

    def video(self, name, content):
        (self.videos / name).write_bytes(content)

    def audit(self, **kwargs):
        return audit_dataset(self.labels, self.videos, self.output, holdout_per_class=1, **kwargs)

    def make_duplicate_dataset(self):
        rows = [
            ["Х-101", "13 секунда", "нарушение"],
            ["Х-102", "26 секунда", "нарушение"],
            ["X-301", "7 секунда", "нарушение"],
            ["Х-201", None, "без нарушения"],
            ["Х-202", None, "без нарушения"],
            ["Х-203", None, "без нарушения"],
        ]
        self.workbook(rows)
        for name, content in [("Х-101.mov", b"positive duplicate"),
                              ("Х-102.mov", b"positive duplicate"),
                              ("X-301.mov", b"positive other"),
                              ("Х-201.mov", b"negative duplicate"),
                              ("Х-202.mov", b"negative duplicate"),
                              ("Х-203.mov", b"negative other")]:
            self.video(name, content)

    def test_duplicate_hashes_cannot_leak_between_partitions(self):
        self.make_duplicate_dataset()
        self.video("unused.mov", b"unlabeled video")
        summary = self.audit()
        self.assertEqual(summary["annotation_rows"], 6)
        self.assertEqual(summary["unique_labeled_videos"], 4)
        self.assertEqual(summary["duplicate_group_count"], 2)
        self.assertEqual(summary["unused_video_files"], ["unused.mov"])
        split = json.loads((self.output / "split.json").read_text(encoding="utf-8"))
        self.assertEqual(len(split["development"]), 2)
        self.assertEqual(len(split["holdout"]), 2)
        self.assertFalse(set(split["development"]) & set(split["holdout"]))
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        annotations = {row["filename"]: row for row in manifest["annotations"]}
        for first, second in (("Х-101.mov", "Х-102.mov"), ("Х-201.mov", "Х-202.mov")):
            self.assertEqual(annotations[first]["sha256"], annotations[second]["sha256"])
            self.assertEqual(annotations[first]["split"], annotations[second]["split"])
        self.assertEqual(annotations["Х-101.mov"]["timestamp_sec"], 13)
        self.assertEqual(annotations["Х-102.mov"]["timestamp_sec"], 26)
        self.assertEqual(annotations["Х-101.mov"]["annotation_kind"], "weak_point")
        self.assertIn("duplicate_timestamps_differ", annotations["Х-101.mov"]["review_flags"])
        self.assertNotIn("start_sec", annotations["Х-101.mov"])
        self.assertNotIn("end_sec", annotations["Х-101.mov"])

    def test_split_stays_stable_if_workbook_order_changes(self):
        self.make_duplicate_dataset()
        self.audit(seed=17)
        first = (self.output / "split.json").read_bytes()
        self.workbook([["Х-203", None, 0], ["Х-202", None, 0], ["Х-201", None, 0],
                       ["X-301", 7, 1], ["Х-102", 26, 1], ["Х-101", 13, 1]])
        self.audit(seed=17)
        self.assertEqual((self.output / "split.json").read_bytes(), first)

    def test_conflicting_duplicate_labels_fail_before_writes(self):
        self.workbook([["Х-101", 13, 1], ["Х-102", None, 0]])
        self.video("Х-101.mov", b"same")
        self.video("Х-102.mov", b"same")
        with self.assertRaisesRegex(DatasetValidationError, "Conflicting binary labels"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_missing_source_and_unknown_label_fail_clearly(self):
        self.workbook([["Х-101", 13, "maybe"]])
        with self.assertRaisesRegex(DatasetValidationError, "row 2.*unknown binary label"):
            self.audit()
        self.workbook([["Х-101", 13, 1]])
        with self.assertRaisesRegex(DatasetValidationError, "Missing source videos"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_exact_alphabet_wins_and_fallback_is_recorded(self):
        self.workbook([["X-301", 11, 1], ["X-302", None, 0]])
        self.video("X-301.mov", b"latin exact")
        self.video("Х-301.mov", b"cyrillic distinct")
        self.video("Х-302.mov", b"cyrillic fallback")
        self.audit()
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        first, second = manifest["annotations"]
        self.assertEqual(first["filename"], "X-301.mov")
        self.assertEqual(first["filename_match"], "exact")
        self.assertEqual(second["filename"], "Х-302.mov")
        self.assertEqual(second["label_filename"], "X-302")
        self.assertIn("confusable_prefix_match", second["review_flags"])

    def test_insufficient_unique_videos_prevent_split(self):
        self.workbook([["Х-101", 13, 1], ["Х-102", 26, 1]])
        self.video("Х-101.mov", b"same")
        self.video("Х-102.mov", b"same")
        with self.assertRaisesRegex(DatasetValidationError, "Class 0 has 0 unique"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_reaudit_preserves_manual_review_columns(self):
        self.make_duplicate_dataset()
        self.audit()
        review_path = self.output / "review.csv"
        with review_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        self.assertTrue(all(row["type"] == "" and row["start_sec"] == "" for row in rows))
        rows[0]["type"] = "manual event type"
        rows[0]["start_sec"] = "12.5"
        with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.audit()
        with review_path.open(encoding="utf-8-sig", newline="") as stream:
            after = list(csv.DictReader(stream))
        self.assertEqual(after[0]["type"], "manual event type")
        self.assertEqual(after[0]["start_sec"], "12.5")


if __name__ == "__main__":
    unittest.main()
