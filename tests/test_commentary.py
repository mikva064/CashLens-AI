"""Local commentary boundaries using synthetic CPU fixtures only."""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import tempfile
import traceback
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from cashlens_ai.backend import LocalQwenBackend, _GpuDutyPacer, _StopAfterJsonObject
from cashlens_ai.commentary import (
    CommentaryError, MAX_COMMENTARY_TOKENS, _PROMPT, _StopAtCommentaryDeadline,
    _SCHEMA, _private_lmfe_logs, _private_prefix_callback, _validate_commentary,
)
from cashlens_ai.video import SampledFrame


class FakeTensor:
    def __init__(self, width):
        self.shape = (1, width)

    def __getitem__(self, key):
        start = key[1].start or 0
        return FakeTensor(max(0, self.shape[1] - start))


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class FakeClock:
    now = 1.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class CommentaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.backend = LocalQwenBackend(
            Path(self.directory.name), gpu_duty_cycle=0.65, gpu_memory_fraction=0.70,
        )
        self.frames = [SampledFrame(t, Image.new("RGB", (32, 32), color))
                       for t, color in ((4.0, "red"), (5.2, "green"), (6.1, "blue"))]
        self.batch = FakeBatch(input_ids=FakeTensor(12))
        self.processor = MagicMock(return_value=self.batch)
        self.processor.apply_chat_template.return_value = "synthetic rendered input"
        self.processor.batch_decode.return_value = [
            '{"d":"Человек в красной одежде перемещает предмет вправо."}',
        ]
        self.model = MagicMock()
        self.model.generate.return_value = FakeTensor(22)
        self.torch = SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            cuda=SimpleNamespace(synchronize=MagicMock(), empty_cache=MagicMock()),
        )
        self.backend._model = self.model
        self.backend._processor = self.processor
        self.backend._torch = self.torch
        self.backend._tokenizer_data = object()
        self.transformers = types.ModuleType("transformers")
        self.transformers.StoppingCriteriaList = list
        self.lmfe = types.ModuleType("lmformatenforcer")
        self.lmfe.CharacterLevelParserConfig = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(alphabet="ASCII", **kw),
        )
        self.lmfe.JsonSchemaParser = MagicMock(return_value=SimpleNamespace(config=None))
        self.integration = types.ModuleType("lmformatenforcer.integrations.transformers")
        self.integration.build_transformers_prefix_allowed_tokens_fn = MagicMock(
            return_value=lambda batch_id, tokens: [1],
        )
        self.modules = {
            "transformers": self.transformers,
            "lmformatenforcer": self.lmfe,
            "lmformatenforcer.integrations": types.ModuleType("lmformatenforcer.integrations"),
            "lmformatenforcer.integrations.transformers": self.integration,
        }

    def call(self, **kw):
        with patch.dict(sys.modules, self.modules):
            return self.backend.comment_event(self.frames, timeout_sec=kw.pop("timeout_sec", 10), **kw)

    def test_reuses_resident_model_trie_lock_and_gpu_policy(self):
        original_trie = self.backend._tokenizer_data
        original_lock = self.backend._lock
        with patch.object(self.backend, "_load_model", wraps=self.backend._load_model) as load:
            first, second = self.call(), self.call()
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "generated")
        self.assertEqual(load.call_count, 2)
        self.assertIs(self.backend._model, self.model)
        self.assertIs(self.backend._lock, original_lock)
        self.assertFalse(original_lock.locked())
        self.assertEqual(self.backend.gpu_memory_fraction, 0.70)
        self.assertEqual(self.model.generate.call_count, 2)
        self.assertEqual(self.torch.cuda.empty_cache.call_count, 2)
        prefix_call = self.integration.build_transformers_prefix_allowed_tokens_fn.call_args
        self.assertIs(prefix_call.args[0], original_trie)
        opts = self.model.generate.call_args.kwargs
        self.assertEqual(opts["max_new_tokens"], 64)
        self.assertFalse(opts["do_sample"])
        self.assertLessEqual(opts["max_time"], 10)
        stop, pacer, deadline = opts["stopping_criteria"]
        self.assertIsInstance(stop, _StopAfterJsonObject)
        self.assertIsInstance(pacer, _GpuDutyPacer)
        self.assertIsInstance(deadline, _StopAtCommentaryDeadline)
        self.assertIs(pacer.cuda, self.torch.cuda)
        self.assertEqual(pacer.duty_cycle, 0.65)
        self.assertEqual(stop.max_new_tokens, 64)
        self.assertEqual(self.processor.call_args.kwargs["max_pixels"], self.backend.max_pixels)
        self.assertEqual(self.processor.call_args.kwargs["images"], [f.image for f in self.frames])

    def test_independent_prompt_only_supplies_ordered_actual_pts_and_images(self):
        self.call()
        content = self.processor.apply_chat_template.call_args.args[0][0]["content"]
        text = " ".join(item.get("text", "") for item in content)
        self.assertIn("среднем кадре f=1", text)
        self.assertIn("Frame f=0, original PTS 4.000000", text)
        self.assertIn("Frame f=1, original PTS 5.200000", text)
        self.assertIn("Frame f=2, original PTS 6.100000", text)
        self.assertNotIn("CATEGORY_DEFINITIONS", text)
        self.assertNotIn("description_template", text)
        self.assertNotIn("cash_split", text)
        self.assertEqual(sum(item["type"] == "image" for item in content), 3)
        self.assertEqual(content[-1], {"type": "text", "text": _PROMPT})
        self.assertEqual(content[0]["text"], "Frame f=0, original PTS 4.000000 seconds:")
        self.assertLess(len(_PROMPT.split()), 110)
        with self.assertRaises(TypeError):
            self.call(rules=[{"id": "NEVER_PASS_CATEGORY"}])

    def test_schema_and_cyrillic_config_are_restored_after_enforcer(self):
        def replace_config(trie, parser):
            parser.config = "LMFE default"
            return lambda batch_id, tokens: [1]
        self.integration.build_transformers_prefix_allowed_tokens_fn.side_effect = replace_config
        self.call()
        call = self.lmfe.JsonSchemaParser.call_args
        schema, config = call.args[0], call.kwargs["config"]
        self.assertEqual(schema["required"], ["d"])
        self.assertEqual(set(schema["properties"]), {"d"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["d"]["maxLength"], 80)
        self.assertEqual(config.max_consecutive_whitespaces, 1)
        self.assertTrue(config.force_json_field_order)
        self.assertIn("Я", config.alphabet)
        self.assertIs(self.lmfe.JsonSchemaParser.return_value.config, config)

    def test_invalid_frames_prevent_model_access(self):
        for times in ([], [1, 2], [1, 2, 3, 4], [1, 1, 2], [3, 2, 1],
                      [-1, 2, 3], [0, float("nan"), 3], [0, True, 2], ["0", 1, 2]):
            with self.subTest(times=times):
                frames = [SimpleNamespace(timestamp_sec=t, image=object()) for t in times]
                with patch.object(self.backend, "_load_model") as load:
                    with self.assertRaises(CommentaryError) as error:
                        self.backend.comment_event(frames, timeout_sec=10)
                self.assertEqual(error.exception.reason_code, "invalid_frames")
                load.assert_not_called()

    def test_timeout_includes_lock_wait(self):
        lock = MagicMock()
        lock.acquire.return_value = False
        self.backend._lock = lock
        with self.assertRaises(CommentaryError) as error:
            self.call(timeout_sec=0.01)
        self.assertEqual(error.exception.reason_code, "timeout")
        self.assertLessEqual(lock.acquire.call_args.kwargs["timeout"], 0.01)
        lock.release.assert_not_called()
        self.model.generate.assert_not_called()

    def test_preprocessing_time_is_subtracted_from_generation_budget(self):
        clock = FakeClock()
        def prepare(**kw):
            clock.now += 3.0
            return self.batch
        self.processor.side_effect = prepare
        with patch("cashlens_ai.commentary.time.monotonic", clock):
            self.call(timeout_sec=10)
        self.assertEqual(self.model.generate.call_args.kwargs["max_time"], 7)

    def test_preprocessing_deadline_prevents_generation(self):
        clock = FakeClock()
        def prepare(**kw):
            clock.now += 11.0
            return self.batch
        self.processor.side_effect = prepare
        with patch("cashlens_ai.commentary.time.monotonic", clock):
            with self.assertRaises(CommentaryError) as error:
                self.call(timeout_sec=10)
        self.assertEqual(error.exception.reason_code, "timeout")
        self.model.generate.assert_not_called()
        self.assertFalse(self.backend._lock.locked())

    def test_deadline_runs_after_pacing_and_late_complete_output_is_rejected(self):
        clock = FakeClock()
        def generate(**kw):
            clock.now += 1.9
            stop = kw["stopping_criteria"]
            with patch("cashlens_ai.backend.time.sleep", clock.sleep):
                stop[1](FakeTensor(13), None)
            self.assertTrue(stop[2](FakeTensor(13), None))
            return FakeTensor(22)
        self.model.generate.side_effect = generate
        with patch("cashlens_ai.commentary.time.monotonic", clock), \
                patch("cashlens_ai.commentary.time.sleep", clock.sleep):
            with self.assertLogs("cashlens.commentary", level="INFO") as logs:
                with self.assertRaises(CommentaryError) as error:
                    self.call(timeout_sec=2)
        self.assertEqual(error.exception.reason_code, "timeout")
        self.assertIn("generated_tokens=10", " ".join(logs.output))

    def test_invalid_timeout_never_touches_model(self):
        for timeout in (0, -1, float("nan"), float("inf"), True, None, "10"):
            with self.subTest(timeout=timeout):
                with self.assertRaises(CommentaryError) as error:
                    self.call(timeout_sec=timeout)
                self.assertEqual(error.exception.reason_code, "invalid_timeout")
        self.model.generate.assert_not_called()

    def test_errors_and_logs_never_contain_private_output_or_path(self):
        private = "PRIVATE_OUTPUT C:/PRIVATE_PATH.mov"
        for raw in ('{"d":"' + private + '"} trailing',
                    '{"d":"Человек украл PRIVATE_OUTPUT"}'):
            self.processor.batch_decode.return_value = [raw]
            with self.assertLogs("cashlens.commentary", level="INFO") as logs:
                with self.assertRaises(CommentaryError) as error:
                    self.call()
            trace = "".join(traceback.format_exception(error.exception))
            self.assertNotIn(private, trace)
            self.assertNotIn("PRIVATE_OUTPUT", " ".join(logs.output))
            self.assertIn("failure_code=", " ".join(logs.output))
        self.model.generate.side_effect = RuntimeError(private)
        with self.assertLogs("cashlens.commentary", level="INFO") as logs:
            with self.assertRaises(CommentaryError) as error:
                self.call()
        self.assertNotIn(private, "".join(traceback.format_exception(error.exception)))
        self.assertNotIn(private, " ".join(logs.output))
        self.assertFalse(self.backend._lock.locked())

    def test_token_limit_invalid_json_is_truncated_and_not_displayed(self):
        self.model.generate.return_value = FakeTensor(12 + MAX_COMMENTARY_TOKENS)
        self.processor.batch_decode.return_value = ['{"d":"PRIVATE_OUTPUT']
        with self.assertRaises(CommentaryError) as error:
            self.call()
        self.assertEqual(error.exception.reason_code, "response_truncated")
        self.assertNotIn("PRIVATE_OUTPUT", str(error.exception))

    def test_success_log_contains_metadata_without_generated_caption(self):
        description = "Человек перемещает PRIVATE_OUTPUT вправо."
        self.processor.batch_decode.return_value = [json.dumps({"d": description})]
        with self.assertLogs("cashlens.commentary", level="INFO") as logs:
            result = self.call()
        self.assertEqual(result["text"], description)
        self.assertNotIn("PRIVATE_OUTPUT", " ".join(logs.output))
        self.assertIn("status=generated failure_code=none", " ".join(logs.output))

    def test_preparation_and_cleanup_failures_cannot_leak_paths_or_poison_lock(self):
        self.processor.side_effect = RuntimeError("PRIVATE_OUTPUT C:/PRIVATE_PATH.mov")
        with patch.object(self.backend, "_memory_diagnostics", side_effect=OSError("PRIVATE_PATH")), \
                patch.object(self.backend, "_release_unused_memory_unlocked", side_effect=OSError("PRIVATE_PATH")):
            with self.assertRaises(CommentaryError) as error:
                self.call()
        self.assertEqual(error.exception.reason_code, "inference_failed")
        self.assertNotIn("PRIVATE", "".join(traceback.format_exception(error.exception)))
        self.assertFalse(self.backend._lock.locked())


class CommentaryValidationTests(unittest.TestCase):
    def test_empty_description_is_uncertain_without_confidence_field(self):
        self.assertEqual(_validate_commentary('{"d":""}'),
                         {"status": "uncertain", "text": ""})
        with self.assertRaises(CommentaryError):
            _validate_commentary('{"s":"uncertain","d":""}')

    def test_strict_json_rejects_duplicates_extra_fields_prose_nonfinite_and_incomplete(self):
        invalid = [
            '{"d":"Человек движется.","d":""}',
            '{"d":"Человек движется.","other":1}',
            '{"d":NaN}', '{"d":Infinity}',
            '```json\n{"d":""}\n```',
            '{"d":""} trailing',
            '{"d":""}{}', '{"d":',
            '[]', 'null', '{"d":true}', '{}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(CommentaryError):
                _validate_commentary(raw)

    def test_bounds_language_whitespace_and_control_characters(self):
        for description in ("Да", "The person moves.", " Человек движется.",
                            "Человек движется. ", "я" * 81, "Человек\nдвижется.",
                            "Человек\u202eдвижется."):
            with self.subTest(description=description), self.assertRaises(CommentaryError):
                _validate_commentary(json.dumps({"d": description}))
        boundary = "Человек " + "я" * 72
        self.assertEqual(len(boundary), 80)
        self.assertEqual(_validate_commentary(json.dumps({"d": boundary}))["text"],
                         boundary)

    def test_unsubstantiated_legal_role_intent_and_links_are_rejected(self):
        for description in (
            "Сотрудник передаёт предмет.", "Клиент берёт предмет.",
            "Человек совершает кражу.", "Человек присваивает деньги.",
            "Человек нарушает статью 188 УК РК.", "Человек виновен.",
            "Человек намеренно прячет предмет.", "Человек забирает предмет себе.",
            "Человек нарушает регламент.", "Описание https://example.invalid",
            "Человек совершает theft.", "Человек идёт <script>.",
        ):
            with self.subTest(description=description):
                with self.assertRaises(CommentaryError) as error:
                    _validate_commentary(json.dumps({"d": description}))
                self.assertEqual(error.exception.reason_code, "unsupported_claim")

    def test_neutral_description_of_synthetic_action_is_allowed(self):
        description = "Человек в красной одежде перемещает предмет вправо."
        self.assertEqual(_validate_commentary(json.dumps({"d": description})),
                         {"status": "generated", "text": description})


class ActualCommentaryGrammarTests(unittest.TestCase):
    def accepts(self, text):
        try:
            from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        except ImportError:
            self.skipTest("lm-format-enforcer unavailable")
        config = CharacterLevelParserConfig(max_consecutive_whitespaces=1, force_json_field_order=True)
        config.alphabet += "".join(chr(code) for code in range(0x0400, 0x0500))
        parser = JsonSchemaParser(_SCHEMA, config=config)
        for char in text:
            if char not in parser.get_allowed_characters():
                return False
            parser = parser.add_character(char)
        return parser.can_end()

    def test_actual_parser_accepts_russian_and_empty_uncertain(self):
        self.assertTrue(self.accepts('{"d":"Человек перемещает предмет вправо."}'))
        self.assertTrue(self.accepts('{"d":""}'))

    def test_actual_parser_bounds_description_and_blocks_other_fields(self):
        self.assertTrue(self.accepts('{"d":"' + "я" * 80 + '"}'))
        self.assertFalse(self.accepts('{"d":"' + "я" * 81 + '"}'))
        self.assertFalse(self.accepts('{"d":"Описание","rule_id":"guess"}'))
        self.assertFalse(self.accepts('{"s":"violation","d":"Описание"}'))


class CommentaryLogPrivacyTests(unittest.TestCase):
    def test_actual_lmfe_fallback_does_not_log_decoded_prefix_or_traceback(self):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData

        data = TokenEnforcerTokenizerData(
            [(0, "x", False), (1, "{", False)], lambda ids: "PRIVATE_OUTPUT", 2, False, 3,
        )
        enforcer = TokenEnforcer(data, JsonSchemaParser(_SCHEMA))
        callback = _private_prefix_callback(lambda batch_id, ids: enforcer.get_allowed_tokens(ids))
        original_filters = tuple(logging.getLogger().filters)
        with patch.object(enforcer, "_collect_allowed_tokens", side_effect=RuntimeError("PRIVATE_OUTPUT")):
            with self.assertLogs(level="DEBUG") as logs:
                result = callback(0, [90])
                logging.debug("safe_test_metadata")
        self.assertEqual(result.allowed_tokens, [2])
        self.assertNotIn("PRIVATE_OUTPUT", " ".join(logs.output))
        self.assertNotIn("Unknown LMFormatEnforcer", " ".join(logs.output))
        self.assertEqual(tuple(logging.getLogger().filters), original_filters)

    def test_actual_lmfe_invalid_character_debug_is_private(self):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData

        data = TokenEnforcerTokenizerData(
            [(0, "x", False), (1, "{", False)],
            lambda ids: "PRIVATE_OUTPUT" if ids else "", 2, False, 3,
        )
        enforcer = TokenEnforcer(data, JsonSchemaParser(_SCHEMA))
        enforcer.get_allowed_tokens([90])
        callback = _private_prefix_callback(lambda batch_id, ids: enforcer.get_allowed_tokens(ids))
        with self.assertLogs(level="DEBUG") as logs:
            result = callback(0, [90, 0])
            logging.debug("safe_test_metadata")
        self.assertIn(2, result.allowed_tokens)
        self.assertNotIn("PRIVATE_OUTPUT", " ".join(logs.output))
        self.assertNotIn("invalid character", " ".join(logs.output))

    def test_filter_preserves_other_threads_modules_levels_and_exception_semantics(self):
        root = logging.getLogger()
        original_level, original_handlers = root.level, tuple(root.handlers)
        path = "/local/site-packages/lmformatenforcer/tokenenforcer.py"
        with self.assertLogs(level="DEBUG") as logs:
            with _private_lmfe_logs():
                for level in (logging.DEBUG, logging.WARNING, logging.ERROR):
                    root.handle(logging.LogRecord("root", level, path, 1, "PRIVATE_OUTPUT", (), None))
                with patch.object(logging, "logThreads", False):
                    root.handle(logging.LogRecord("root", logging.ERROR, path, 1,
                                                  "PRIVATE_OUTPUT", (), None))
                def other_thread():
                    root.handle(logging.LogRecord("root", logging.WARNING, path, 1,
                                                  "other_thread_unaffected", (), None))
                worker = threading.Thread(target=other_thread)
                worker.start()
                worker.join()
                logging.warning("other_module_unaffected")
        self.assertNotIn("PRIVATE_OUTPUT", " ".join(logs.output))
        self.assertIn("other_thread_unaffected", " ".join(logs.output))
        self.assertIn("other_module_unaffected", " ".join(logs.output))
        self.assertEqual(root.level, original_level)
        self.assertEqual(tuple(root.handlers), original_handlers)
        failure = RuntimeError("synthetic callback failure")
        callback = _private_prefix_callback(MagicMock(side_effect=failure))
        with self.assertRaises(RuntimeError) as error:
            callback(0, [])
        self.assertIs(error.exception, failure)
        self.assertFalse(any(type(guard).__name__ == "_CaptionLmfeLogFilter" for guard in root.filters))


if __name__ == "__main__":
    unittest.main()
