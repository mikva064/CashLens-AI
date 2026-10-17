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
