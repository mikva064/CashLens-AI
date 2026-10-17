import base64
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from cashlens_ai.contracts import AnalysisResult, EvidenceFrame, RuleSource, Violation
from cashlens_ai.errors import CashLensError
from cashlens_ai.integration import to_api_payload


class IntegrationTests(unittest.TestCase):
    def test_clean_response_matches_jury_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = to_api_payload(AnalysisResult(verdict="no_violation", violations=[]),
                                     job_id="test-job", evidence_root=tmp)
            self.assertEqual(payload, {"job_id": "test-job", "status": "done",
                                      "verdict": "no_violation", "violations": []})

    def _result(self, image_path):
        return AnalysisResult(verdict="violation", violations=[Violation(
            type="Synthetic test", timecode_sec=1, description="Synthetic red square",
            rule_id="test_rule", source=RuleSource(document="Synthetic test source",
            section="test", url="https://example.invalid/test"),
            evidence_frames=[EvidenceFrame(timestamp_sec=1, image_path=image_path)])])

    def test_valid_jpeg_encoded_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.jpg"
            Image.new("RGB", (8, 8), "red").save(path, format="JPEG")
            event = to_api_payload(self._result(path), job_id="test", evidence_root=tmp)["violations"][0]
            self.assertEqual(base64.b64decode(event["evidence_frames"][0]["image_base64"]), path.read_bytes())
            self.assertIn("https://example.invalid/test", event["description"])
            self.assertEqual(set(event), {"type", "timecode_sec", "description", "evidence_frames"})

    def test_outside_job_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "job"
            root.mkdir()
            path = Path(tmp) / "private.jpg"
            Image.new("RGB", (8, 8)).save(path)
            with self.assertRaises(CashLensError):
                to_api_payload(self._result(path), job_id="test", evidence_root=root)

    def test_non_jpeg_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            Image.new("RGB", (8, 8)).save(path)
            with self.assertRaises(CashLensError):
                to_api_payload(self._result(path), job_id="test", evidence_root=tmp)
