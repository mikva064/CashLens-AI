import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from urllib.parse import unquote

from cashlens_ai.errors import DatasetValidationError
from cashlens_ai.review import _IMPORT_VALIDATOR_JS, build_review


def payload_from(html):
    match = re.search(r'<script type="application/json" id="cashlens-data">(.*?)</script>', html, re.S)
    return json.loads(match.group(1))


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        self.manifest_path = self.root / "manifest.json"
        self.output = self.root / "local" / "review.html"
        self.development_sha = "a" * 64
        self.holdout_sha = "b" * 64
        self.unused_sha = "c" * 64
        self.manifest = {
            "schema_version": 1,
            "source": {"videos_dir": str(self.videos)},
            "groups": [
                {"sha256": self.development_sha, "filenames": ["X-4.mov", "Х-47.mov"],
                 "split": "development", "label": 1,
                 "review_flags": ["duplicate_content", "duplicate_timestamps_differ"]},
                {"sha256": self.holdout_sha, "filenames": ["secret-holdout.mov"],
                 "split": "holdout", "label": 0},
                {"sha256": self.unused_sha, "filenames": ["unused.mov"],
                 "split": "unused", "label": None},
            ],
            "annotations": [
                {"source_row": 2, "sha256": self.development_sha, "filename": "X-4.mov",
                 "label_filename": "X-4", "timestamp_raw": "57 секунда", "timestamp_sec": 57,
                 "annotation_kind": "weak_point"},
                {"source_row": 3, "sha256": self.development_sha, "filename": "Х-47.mov",
                 "label_filename": "Х-47", "timestamp_raw": "1минута 8 секунда", "timestamp_sec": 68,
                 "annotation_kind": "weak_point"},
                {"source_row": 4, "sha256": self.holdout_sha, "filename": "secret-holdout.mov",
                 "label_filename": "secret-holdout", "timestamp_raw": "secret-point-text",
                 "timestamp_sec": None, "annotation_kind": "video_label_only"},
            ],
        }

    def build(self, **kwargs):
        self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8")
        result = build_review(self.manifest_path, self.output, **kwargs)
        self.assertEqual(result, self.output.resolve())
        return result.read_text(encoding="utf-8")

    def test_default_contains_only_development_unique_groups(self):
        html = self.build()
        self.assertNotIn("secret-holdout.mov", html)
        self.assertNotIn(self.holdout_sha, html)
        self.assertNotIn("secret-point-text", html)
        self.assertNotIn("unused.mov", html)
        payload = payload_from(html)
        self.assertFalse(payload["includes_holdout"])
        self.assertEqual(len(payload["videos"]), 1)
        video = payload["videos"][0]
        self.assertEqual(video["sha256"], self.development_sha)
        self.assertEqual([row["timestamp_sec"] for row in video["source_annotations"]], [57, 68])
        self.assertNotIn("start_sec", video["source_annotations"][0])

    def test_explicit_holdout_mode_includes_holdout_but_not_unused(self):
        payload = payload_from(self.build(include_holdout=True))
        self.assertTrue(payload["includes_holdout"])
        self.assertEqual({video["sha256"] for video in payload["videos"]},
                         {self.development_sha, self.holdout_sha})

    def test_exact_filenames_and_original_file_urls_are_preserved(self):
        html = self.build()
        video = payload_from(html)["videos"][0]
        self.assertEqual(video["filenames"], ["X-4.mov", "Х-47.mov"])
        self.assertEqual(video["video_urls"], [(self.videos / name).as_uri() for name in video["filenames"]])
        self.assertIn("Х-47.mov", unquote(video["video_urls"][1]))
        self.assertTrue(all(url.startswith("file:///") for url in video["video_urls"]))
        self.assertEqual(list(self.videos.iterdir()), [])  # no media copied or created

    def test_untrusted_values_cannot_end_data_script_or_inject_markup(self):
        malicious = '</script><script>alert("injected")</script>&\u2028\u2029'
        self.manifest["annotations"][0]["timestamp_raw"] = malicious
        self.manifest["annotations"][0]["label_filename"] = '<img src=x onerror="alert(1)">'
        self.manifest["groups"][0]["filenames"][0] = 'X-<img onerror="bad">.mov'
        html = self.build()
        self.assertNotIn(malicious, html)
        self.assertNotIn('<img src=x onerror="alert(1)">', html)
        self.assertNotIn('X-<img onerror="bad">.mov', html)
        self.assertIn(r'\u003c/script\u003e', html)
        self.assertIn(r'\u0026\u2028\u2029', html)
        payload = payload_from(html)
        self.assertEqual(payload["videos"][0]["source_annotations"][0]["timestamp_raw"], malicious)
        self.assertEqual(len(re.findall(r'<script(?:\s|>)', html)), 2)

    def test_viewer_has_offline_multi_event_and_local_export_controls(self):
        html = self.build()
        self.assertIn("connect-src 'none'", html)
        self.assertIn('id="event-template"', html)
        self.assertIn('id="add-event"', html)
        self.assertIn('data-field="start_sec"', html)
        self.assertIn('data-field="end_sec"', html)
        self.assertIn('data-field="rule_id"', html)
        self.assertIn('data-field="evidence_sec"', html)
        self.assertIn("URL.createObjectURL(new Blob", html)
        self.assertIn('id="import-file" type="file" accept=".json,application/json"', html)
        self.assertIn("JSON.parse(await file.text())", html)
        self.assertIn("const updates = validateReviewImport(candidate, state)", html)
        self.assertIn("dirty && !window.confirm(", html)
        self.assertIn("Нормативные ссылки ещё не прошли правовую проверку", html)
        self.assertNotIn("fetch(", html)
        self.assertNotIn("XMLHttpRequest", html)
        self.assertNotRegex(html, r'(?:src|href)=["\']https?://')

    def test_bad_manifest_or_path_fails_before_output(self):
        self.manifest["groups"][0]["filenames"] = ["../outside.mov"]
        with self.assertRaisesRegex(DatasetValidationError, "basenames"):
            self.build()
        self.assertFalse(self.output.exists())
        self.manifest = []
        with self.assertRaisesRegex(DatasetValidationError, "JSON object"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_invalid_point_is_not_silently_embedded(self):
        self.manifest["annotations"][0]["timestamp_sec"] = -2
        with self.assertRaisesRegex(DatasetValidationError, "Invalid weak timestamp"):
            self.build()
        self.assertFalse(self.output.exists())


@unittest.skipUnless(shutil.which("node"), "Node.js is needed for browser-side import validation tests")
class ImportValidationTests(unittest.TestCase):
    """Exercise the actual standalone JS validator, without a browser or video IO."""

    def setUp(self):
        self.sha = "a" * 64
        self.videos = [{"sha256": self.sha, "label": 1, "filenames": ["Х-4.mov"],
                        "notes": "current unsaved notes", "events": []}]
        self.candidate = {
            "schema_version": 1, "purpose": "manual_review", "includes_holdout": False,
            "videos": [{"sha256": self.sha, "binary_label": 1, "reviewed": True,
                        "notes": "restored video notes", "events": [{
                            "event_id": "event-7", "type": "manual type", "start_sec": 55.5,
                            "end_sec": 68, "rule_id": "PENDING", "evidence_sec": [57, 65.2],
                            "notes": "restored event notes"}]}],
        }

    def validate(self, *, mutation=""):
        script = _IMPORT_VALIDATOR_JS + r'''
const fs = require('node:fs');
const {candidate, videos} = JSON.parse(fs.readFileSync(0, 'utf8'));
const before = JSON.stringify(videos);
''' + mutation + r'''
try {
  const updates = validateReviewImport(candidate, videos);
  console.log(JSON.stringify({ok: true, updates, unchanged: JSON.stringify(videos) === before}));
} catch (error) {
  console.log(JSON.stringify({ok: false, error: error.message, unchanged: JSON.stringify(videos) === before}));
}
'''
        result = subprocess.run([shutil.which("node"), "-e", script],
                                input=json.dumps({"candidate": self.candidate, "videos": self.videos}),
                                text=True, encoding="utf-8", capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_exported_fields_restore_without_mutating_current_state(self):
        result = self.validate()
        self.assertTrue(result["ok"])
        self.assertTrue(result["unchanged"])
        update = result["updates"][0]
        self.assertEqual(update["index"], 0)
        self.assertTrue(update["reviewed"])
        self.assertEqual(update["notes"], "restored video notes")
        event = update["events"][0]
        for key, value in self.candidate["videos"][0]["events"][0].items():
            self.assertEqual(event[key], value)
        self.assertEqual(event["evidence_input"], "57, 65.2")

    def test_draft_events_with_blank_boundaries_roundtrip(self):
        event = self.candidate["videos"][0]["events"][0]
        event.update(type="", start_sec=None, end_sec=None, rule_id="", evidence_sec=[], notes="")
        result = self.validate()
        self.assertTrue(result["ok"])
        self.assertIsNone(result["updates"][0]["events"][0]["start_sec"])

    def test_unknown_hash_after_valid_record_rejects_entire_import(self):
        self.candidate["videos"].append({"sha256": "b" * 64})
        result = self.validate()
        self.assertFalse(result["ok"])
        self.assertIn("неизвестный SHA256", result["error"])
        self.assertTrue(result["unchanged"])
        self.assertNotIn("updates", result)

    def test_malformed_intervals_nonfinite_points_and_schema_are_rejected(self):
        mutations = [
            "candidate.schema_version = 2;",
            "candidate.videos[0].reviewed = 'yes';",
            "candidate.videos[0].binary_label = 0;",
            "candidate.videos[0].events[0].end_sec = 1;",
            "candidate.videos[0].events[0].start_sec = Infinity;",
            "candidate.videos[0].events[0].evidence_sec = [NaN];",
            "candidate.videos[0].events[0].evidence_sec = [-1];",
            "candidate.videos[0].events[0].evidence_sec = ['57'];",
            "candidate.videos[0].events.push(candidate.videos[0].events[0]);",
            "candidate.videos.push(candidate.videos[0]);",
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                result = self.validate(mutation=mutation)
                self.assertFalse(result["ok"])
                self.assertTrue(result["unchanged"])


if __name__ == "__main__":
    unittest.main()
