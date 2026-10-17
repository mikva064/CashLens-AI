"""Contract and registry validation with synthetic local JSON only."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from cashlens_ai.contracts import AnalysisResult, EvidenceFrame, Rule, RuleSource, Violation
from cashlens_ai.errors import RuleValidationError
from cashlens_ai.rules import load_rules


SOURCE = {
    "document": "Synthetic reviewed procedure",
    "section": "Example section 1",
    "url": "https://example.invalid/reviewed-procedure",
}


def reviewed_rule(**overrides):
    values = {
        "id": "synthetic_rule",
        "title": "Synthetic event",
        "observable_condition": "A test object is visibly moved into the marked box.",
        "source": dict(SOURCE),
        "verified": True,
    }
    values.update(overrides)
    return values


def violation(**overrides):
    values = {
        "type": "Synthetic event",
        "timecode_sec": 12.5,
        "description": "The employee visibly places the test object in the marked box.",
        "rule_id": "synthetic_rule",
        "source": dict(SOURCE),
        "evidence_frames": [{"timestamp_sec": 12.5, "image_path": "evidence/frame.jpg"}],
    }
    values.update(overrides)
    return Violation.model_validate(values)


class ResultContractTests(unittest.TestCase):
    def test_positive_result_keeps_reviewed_source_and_source_frame_timestamp(self):
        event = violation()
        result = AnalysisResult(verdict="violation", violations=[event])
        decoded = json.loads(result.model_dump_json())
        self.assertEqual(decoded["violations"][0]["source"], SOURCE)
        self.assertEqual(decoded["violations"][0]["timecode_sec"], 12.5)
        self.assertEqual(decoded["violations"][0]["evidence_frames"][0]["timestamp_sec"], 12.5)

    def test_negative_result_requires_no_events(self):
        result = AnalysisResult(verdict="no_violation", violations=[])
        self.assertEqual(result.violations, [])
        with self.assertRaises(ValidationError):
            AnalysisResult(verdict="no_violation", violations=[violation()])

    def test_positive_verdict_requires_at_least_one_event(self):
        with self.assertRaises(ValidationError):
            AnalysisResult(verdict="violation", violations=[])

    def test_source_metadata_is_required_and_not_freeform(self):
        for missing in ("document", "section", "url"):
            with self.subTest(missing=missing):
                source = dict(SOURCE)
                source.pop(missing)
                with self.assertRaises(ValidationError):
                    RuleSource.model_validate(source)
        with self.assertRaises(ValidationError):
            RuleSource.model_validate({**SOURCE, "model_invented_clause": "unreviewed"})

    def test_sources_require_web_urls(self):
        for url in ("file:///private/rule.txt", "ftp://example.invalid/rule", "not a URL"):
            with self.subTest(url=url), self.assertRaises(ValidationError):
                RuleSource.model_validate({**SOURCE, "url": url})

    def test_source_urls_reject_broken_hosts_and_embedded_whitespace(self):
        for url in (
            "https:// bad.example/rule",
            "https://example.invalid/white space",
            "https://example.invalid\t/rule",
            "https://[broken-ipv6]/rule",
            "https://?query=missing-host",
        ):
            with self.subTest(url=url), self.assertRaises(ValidationError):
                RuleSource.model_validate({**SOURCE, "url": url})

    def test_whitespace_only_source_metadata_is_rejected(self):
        for field in ("document", "section"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                RuleSource.model_validate({**SOURCE, field: " \t\n "})

    def test_whitespace_only_event_descriptions_and_titles_are_rejected(self):
        with self.assertRaises(ValidationError):
            violation(description=" \t\n ")
        with self.assertRaises(ValidationError):
            violation(type=" \t\n ")
        with self.assertRaises(ValidationError):
            Rule.model_validate(reviewed_rule(title=" \t\n "))
        with self.assertRaises(ValidationError):
            Rule.model_validate(reviewed_rule(observable_condition=" " * 20))

    def test_surrounding_whitespace_is_trimmed_before_storage(self):
        source = RuleSource.model_validate({
            "document": "  Reviewed source  ",
            "section": "  Section 1\t",
            "url": "  https://example.invalid/rule  ",
        })
        self.assertEqual(source.document, "Reviewed source")
        self.assertEqual(source.section, "Section 1")
        self.assertEqual(source.url, "https://example.invalid/rule")
        event = violation(description="  Visible movement. \n")
        self.assertEqual(event.description, "Visible movement.")

    def test_evidence_requires_one_to_three_frames(self):
        frame = {"timestamp_sec": 12.5, "image_path": "evidence/frame.jpg"}
        for frames in ([], [frame] * 4):
            with self.subTest(count=len(frames)), self.assertRaises(ValidationError):
                violation(evidence_frames=frames)

    def test_timestamps_reject_negative_and_nonfinite_numbers(self):
        for timestamp in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValidationError):
                    EvidenceFrame(timestamp_sec=timestamp, image_path=Path("evidence/frame.jpg"))
                with self.assertRaises(ValidationError):
                    violation(timecode_sec=timestamp)

    def test_unknown_result_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            AnalysisResult.model_validate({
                "verdict": "no_violation", "violations": [], "confidence": 1.0,
            })


class RuleRegistryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "rules.json"

    def write(self, data):
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return self.path

    def test_valid_registry_roundtrips_source_and_verification(self):
        loaded = load_rules(self.write([reviewed_rule()]))
        self.assertEqual(len(loaded), 1)
        self.assertTrue(loaded[0].verified)
        self.assertEqual(loaded[0].source.model_dump(mode="json"), SOURCE)

    def test_empty_registry_is_an_error(self):
        with self.assertRaisesRegex(RuleValidationError, "empty"):
            load_rules(self.write([]))

    def test_duplicate_rule_ids_are_an_error(self):
        with self.assertRaisesRegex(RuleValidationError, "Duplicate"):
            load_rules(self.write([reviewed_rule(), reviewed_rule(title="Second condition")]))

    def test_unverified_rule_blocks_entire_registry(self):
        values = [reviewed_rule(), reviewed_rule(id="second_rule", verified=False)]
        with self.assertRaisesRegex(RuleValidationError, "verified"):
            load_rules(self.write(values))

    def test_omitted_verification_defaults_to_unverified(self):
        values = reviewed_rule()
        values.pop("verified")
        self.assertFalse(Rule.model_validate(values).verified)
        with self.assertRaisesRegex(RuleValidationError, "verified"):
            load_rules(self.write([values]))

    def test_missing_or_malformed_registry_is_an_error(self):
        with self.assertRaises(RuleValidationError):
            load_rules(self.path)
        self.path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(RuleValidationError):
            load_rules(self.path)

    def test_missing_source_and_unsupported_fields_are_an_error(self):
        invalid = reviewed_rule()
        invalid.pop("source")
        with self.assertRaises(RuleValidationError):
            load_rules(self.write([invalid]))
        invalid = reviewed_rule(autogenerated_legal_text="Not reviewed")
        with self.assertRaises(RuleValidationError):
            load_rules(self.write([invalid]))


if __name__ == "__main__":
    unittest.main()
