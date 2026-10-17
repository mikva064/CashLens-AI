import json
import unittest

from cashlens_ai.errors import InvalidModelOutputError
from cashlens_ai.pipeline import parse_model_response


class LocalModelFormattingTests(unittest.TestCase):
    def parse(self, text):
        return parse_model_response(text, allowed_ids={"test_rule"}, first_sec=0, last_sec=2)

    def test_complete_object_with_one_stray_quote(self):
        self.assertEqual(self.parse('{"events":[]}"'), [])

    def test_complete_fenced_object_can_stop_before_closing_fence(self):
        self.assertEqual(self.parse('```json\n{"events":[]}'), [])
        self.assertEqual(self.parse('```json\n{"events":[]}\n```'), [])

    def test_prose_second_object_or_truncation_remain_errors(self):
        for text in ('{"events":[]} fabricated evidence', '{"events":[]}{"events":[]}',
                     '{"events":[', '[]', '{"events":[]}""', 'reasoning {"events":[]}'):
            with self.subTest(text=text), self.assertRaises(InvalidModelOutputError):
                self.parse(text)

    def test_formatting_does_not_bypass_rule_or_time_validation(self):
        for event in ('{"rule_id":"unknown","timecode_sec":1,"description":"test"}',
                      '{"rule_id":"test_rule","timecode_sec":7,"description":"test"}'):
            with self.assertRaises(InvalidModelOutputError):
                self.parse('{"events":[' + event + ']}"')

    def test_fine_proof_requires_exact_ordered_frames_and_candidate_focus(self):
        valid = {"rule_id": "test_rule", "timecode_sec": 1.0, "description": "Test evidence",
                 "evidence_timestamps_sec": [0.0, 1.0, 2.0], "cash_context_timestamp_sec": 0.0}

        def parse_fine(event):
            return parse_model_response(
                json.dumps({"events": [event]}), allowed_ids={"test_rule"}, first_sec=0, last_sec=2,
                require_proof=True, frame_timestamps=[0.0, 0.5, 1.0, 1.5, 2.0],
                candidate_timecode_sec=1.0, candidate_radius_sec=0.25,
            )

        self.assertEqual(parse_fine(valid)[0].evidence_timestamps_sec, [0.0, 1.0, 2.0])
        missing = {key: value for key, value in valid.items() if key != "evidence_timestamps_sec"}
        for event in (missing,
                      {**valid, "evidence_timestamps_sec": [0.0, 1.0, 1.0]},
                      {**valid, "evidence_timestamps_sec": [2.0, 1.0, 0.0]},
                      {**valid, "evidence_timestamps_sec": [0.1, 1.0, 2.0]},
                      {**valid, "evidence_timestamps_sec": [False, 1.0, 2.0]},
                      {**valid, "cash_context_timestamp_sec": 0.5},
                      {**valid, "cash_context_timestamp_sec": False},
                      {**valid, "timecode_sec": 1.5},
                      {**valid, "timecode_sec": 1.5,
                       "evidence_timestamps_sec": [0.0, 1.5, 2.0]}):
            with self.subTest(event=event), self.assertRaises(InvalidModelOutputError):
                parse_fine(event)
        self.assertEqual(parse_model_response(
            '{"events":[]}', allowed_ids={"test_rule"}, first_sec=0, last_sec=2,
            require_proof=True, frame_timestamps=[0.0, 1.0, 2.0],
            candidate_timecode_sec=1, candidate_radius_sec=0.25,
        ), [])
        with self.assertRaises(InvalidModelOutputError):
            parse_model_response(
                json.dumps({"events": [valid] * 4}), allowed_ids={"test_rule"}, first_sec=0, last_sec=2,
                require_proof=True, frame_timestamps=[0.0, 1.0, 2.0],
                candidate_timecode_sec=1, candidate_radius_sec=0.25,
            )
