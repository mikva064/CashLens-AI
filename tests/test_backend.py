"""Backend boundaries without weights, network access or a CUDA installation."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cashlens_ai.backend import (COARSE_DESCRIPTION, DESCRIPTION_MAX_CHARS, MAX_WINDOW_EVENTS,
                                 LocalQwenBackend, _StopAfterJsonObject, _build_output_schema,
                                 _build_prompt, _has_complete_json_object)
from cashlens_ai.errors import ModelUnavailableError, RuleValidationError


RULES = [{
    "id": "demo_rule",
    "title": "Reviewed example",
    "observable_condition": "The employee visibly places the test object in the marked box.",
    "source": {"document": "Demonstration only", "section": "1", "url": "https://example.invalid"},
}]


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.backend = LocalQwenBackend(Path(self.directory.name))
        self.frames = [
            SimpleNamespace(timestamp_sec=10.25, image=object()),
            SimpleNamespace(timestamp_sec=11.5, image=object()),
        ]

    def fake_dependencies(self, *, cuda=True, bf16=True):
        torch = types.ModuleType("torch")
        torch.bfloat16 = "bf16"
        torch.float16 = "fp16"
        torch.cuda = SimpleNamespace(
            is_available=lambda: cuda,
            is_bf16_supported=lambda: bf16,
            empty_cache=MagicMock(),
        )
        torch.inference_mode = contextlib.nullcontext
        transformers = types.ModuleType("transformers")
        model = MagicMock()
        model.eval.return_value = model
        transformers.Qwen3VLForConditionalGeneration = SimpleNamespace(
            from_pretrained=MagicMock(return_value=model)
        )
        processor = MagicMock()
        batch = FakeBatch(input_ids=SimpleNamespace(shape=(1, 99)), attention_mask=object())
        processor.return_value = batch
        processor.batch_decode.return_value = ['{"events":[]}']
        transformers.AutoProcessor = SimpleNamespace(from_pretrained=MagicMock(return_value=processor))
        transformers.BitsAndBytesConfig = MagicMock(return_value="quant-config")
        transformers.StoppingCriteriaList = list
        lmfe = types.ModuleType("lmformatenforcer")
        lmfe.JsonSchemaParser = MagicMock(return_value=SimpleNamespace(config=None))
        lmfe.CharacterLevelParserConfig = MagicMock(
            side_effect=lambda **kwargs: SimpleNamespace(alphabet="ASCII", **kwargs)
        )
        integration = types.ModuleType("lmformatenforcer.integrations.transformers")
        integration.build_token_enforcer_tokenizer_data = MagicMock(return_value="tokenizer-trie")
        integration.build_transformers_prefix_allowed_tokens_fn = MagicMock(
            return_value=lambda batch_id, tokens: [1]
        )
        transformers._lmfe = lmfe
        transformers._lmfe_integration = integration
        return torch, transformers, model, processor, batch

    @staticmethod
    def dependency_modules(torch, transformers):
        return {"torch": torch, "transformers": transformers,
                "lmformatenforcer": transformers._lmfe,
                "lmformatenforcer.integrations": types.ModuleType("lmformatenforcer.integrations"),
                "lmformatenforcer.integrations.transformers": transformers._lmfe_integration}

    def test_missing_rules_prevent_model_loading(self):
        with patch.object(self.backend, "_load_model") as load:
            with self.assertRaises(RuleValidationError):
                self.backend.describe(self.frames, [])
        load.assert_not_called()

    def test_duplicate_rule_ids_prevent_model_loading(self):
        with patch.object(self.backend, "_load_model") as load:
            with self.assertRaises(RuleValidationError):
                self.backend.describe(self.frames, RULES + RULES)
        load.assert_not_called()

    def test_invalid_timestamps_prevent_model_loading(self):
        for values in ([1.0, 0.5], [float("nan")], [-0.1]):
            with self.subTest(values=values), patch.object(self.backend, "_load_model") as load:
                frames = [SimpleNamespace(timestamp_sec=value, image=object()) for value in values]
                with self.assertRaises(ValueError):
                    self.backend.describe(frames, RULES)
                load.assert_not_called()

    def test_offline_generation_preserves_absolute_times_and_raw_response(self):
        torch, transformers, model, processor, batch = self.fake_dependencies()
        raw = '  {"events":[]}  '
        processor.batch_decode.return_value = [raw]
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            result = self.backend.describe(self.frames, RULES, timeout_sec=7.5)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(result, raw)
        self.assertEqual(batch.device, "cuda:0")
        options = transformers.Qwen3VLForConditionalGeneration.from_pretrained.call_args.kwargs
        self.assertTrue(options["local_files_only"])
        self.assertFalse(options["trust_remote_code"])
        self.assertTrue(options["use_safetensors"])
        self.assertEqual(options["device_map"], {"": 0})
        self.assertEqual(options["attn_implementation"], "sdpa")
        self.assertEqual(options["quantization_config"], "quant-config")
        model.to.assert_not_called()
        self.assertEqual(model.generate.call_args.kwargs["max_time"], 7.5)
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])
        stopping = model.generate.call_args.kwargs["stopping_criteria"]
        self.assertEqual(len(stopping), 1)
        self.assertIsInstance(stopping[0], _StopAfterJsonObject)
        self.assertEqual(stopping[0].prompt_length, 99)
        self.assertEqual(stopping[0].max_new_tokens, self.backend.max_new_tokens)
        self.assertIs(
            model.generate.call_args.kwargs["prefix_allowed_tokens_fn"],
            transformers._lmfe_integration.build_transformers_prefix_allowed_tokens_fn.return_value,
        )
        schema_call = transformers._lmfe.JsonSchemaParser.call_args
        schema = schema_call.args[0]
        self.assertEqual(schema["properties"]["events"]["maxItems"], 3)
        event_properties = schema["properties"]["events"]["items"]["properties"]
        self.assertEqual(event_properties["rule_id"]["enum"], ["demo_rule"])
        self.assertEqual(event_properties["timecode_sec"]["enum"], [10.25, 11.5])
        self.assertEqual(event_properties["description"]["enum"], [COARSE_DESCRIPTION])
        self.assertIn("Я", schema_call.kwargs["config"].alphabet)
        self.assertTrue(schema_call.kwargs["config"].force_json_field_order)
        self.assertIs(transformers._lmfe.JsonSchemaParser.return_value.config,
                      schema_call.kwargs["config"])
        chat = processor.apply_chat_template.call_args.args[0]
        all_text = " ".join(item.get("text", "") for item in chat[0]["content"])
        self.assertIn("10.250000 seconds", all_text)
        self.assertIn("11.500000 seconds", all_text)
        self.assertIn('"id": "demo_rule"', all_text)
        self.assertIn("Never identify people", all_text)
        self.assertIn("Do not invent laws", all_text)
        self.assertIn("First establish the visible scene", all_text)
        self.assertIn("transaction sequence needed to satisfy EVERY", all_text)
        self.assertIn('evidence is missing, return {"events":[]}', all_text)
        self.assertEqual(processor.call_args.kwargs["images"], [frame.image for frame in self.frames])

    def test_cuda_absence_raises_error_without_cpu_or_empty_result(self):
        torch, transformers, *_ = self.fake_dependencies(cuda=False)
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            with self.assertRaisesRegex(ModelUnavailableError, "CUDA GPU"):
                self.backend.describe(self.frames, RULES)
        transformers.Qwen3VLForConditionalGeneration.from_pretrained.assert_not_called()

    def test_fp16_and_unquantized_option(self):
        self.backend = LocalQwenBackend(Path(self.directory.name), quantize_4bit=False)
        torch, transformers, *_ = self.fake_dependencies(bf16=False)
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            self.backend.describe(self.frames, RULES)
        options = transformers.Qwen3VLForConditionalGeneration.from_pretrained.call_args.kwargs
        self.assertEqual(options["dtype"], "fp16")
        self.assertNotIn("quantization_config", options)

    def test_load_error_is_not_reported_as_no_violation(self):
        torch, transformers, *_ = self.fake_dependencies()
        transformers.Qwen3VLForConditionalGeneration.from_pretrained.side_effect = RuntimeError("private path")
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            with self.assertRaises(ModelUnavailableError) as error:
                self.backend.describe(self.frames, RULES)
        self.assertNotIn("private path", str(error.exception))

    def test_empty_generation_is_an_error(self):
        torch, transformers, _, processor, _ = self.fake_dependencies()
        processor.batch_decode.return_value = [""]
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            with self.assertRaisesRegex(ModelUnavailableError, "no usable text"):
                self.backend.describe(self.frames, RULES)

    def test_model_is_reused_and_close_releases_references(self):
        torch, transformers, *_ = self.fake_dependencies()
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            self.backend.describe(self.frames, RULES)
            self.backend.describe(self.frames, RULES)
            self.backend.close()
        self.assertEqual(transformers.Qwen3VLForConditionalGeneration.from_pretrained.call_count, 1)
        self.assertIsNone(self.backend._model)
        self.assertIsNone(self.backend._processor)
        self.assertIsNone(self.backend._tokenizer_data)
        transformers._lmfe_integration.build_token_enforcer_tokenizer_data.assert_called_once()
        torch.cuda.empty_cache.assert_called_once()

    def test_enforcer_is_required_without_unconstrained_fallback(self):
        torch, transformers, model, *_ = self.fake_dependencies()
        modules = self.dependency_modules(torch, transformers)
        modules["lmformatenforcer.integrations.transformers"] = None
        with patch.dict(sys.modules, modules), patch.dict(os.environ):
            with self.assertRaises(ModelUnavailableError):
                self.backend.describe(self.frames, RULES)
        model.generate.assert_not_called()

    def test_trie_is_reused_but_schema_parser_is_new_for_each_window(self):
        torch, transformers, *_ = self.fake_dependencies()
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            self.backend.describe(self.frames, RULES)
            later_frames = [SimpleNamespace(timestamp_sec=20.5, image=object())]
            self.backend.describe(later_frames, RULES, phase="fine")
        transformers._lmfe_integration.build_token_enforcer_tokenizer_data.assert_called_once()
        self.assertEqual(transformers._lmfe.JsonSchemaParser.call_count, 2)
        schema = transformers._lmfe.JsonSchemaParser.call_args.args[0]
        times = schema["properties"]["events"]["items"]["properties"]["timecode_sec"]
        self.assertEqual(times["enum"], [20.5])
        self.assertEqual(schema["properties"]["events"]["maxItems"], 1)
        description = schema["properties"]["events"]["items"]["properties"]["description"]
        self.assertEqual(description["maxLength"], 96)
        self.assertNotIn("enum", description)

    def test_generation_diagnostics_contain_only_safe_metadata(self):
        torch, transformers, model, processor, _ = self.fake_dependencies()
        processor.batch_decode.return_value = ['{"events":[{"description":"PRIVATE OUTPUT"}']
        model.generate.return_value.__getitem__.return_value.shape = (1, 384)
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            with self.assertLogs("cashlens.model", level="INFO") as logs:
                self.backend.describe(self.frames, RULES)
        diagnostic = " ".join(logs.output)
        self.assertIn("phase=coarse nframes=2 generated_tokens=384", diagnostic)
        self.assertIn("complete_json=False", diagnostic)
        self.assertIn("token_limit=True", diagnostic)
        self.assertIn("latency_sec=", diagnostic)
        self.assertNotIn("PRIVATE OUTPUT", diagnostic)
        self.assertNotIn(self.directory.name, diagnostic)

    def test_prompts_use_bounded_phase_specific_descriptions(self):
        coarse = _build_prompt(RULES, "coarse")
        fine = _build_prompt(RULES, "fine")
        self.assertIn(COARSE_DESCRIPTION, coarse)
        self.assertIn("Do not narrate or explain candidates", coarse)
        self.assertIn("at most ONE event", fine)
        self.assertIn("at most 12 words and 96 characters", fine)
        self.assertNotIn(COARSE_DESCRIPTION, fine)


class JsonStoppingTests(unittest.TestCase):
    def test_complete_first_object_ends_generation(self):
        for text in ('{"events":[]}', ' \n{"events":[]}  ', '{"wrong_schema": true}',
                     '{"events":[]}"', '{"events":[]} {"events":[1]}'):
            with self.subTest(text=text):
                self.assertTrue(_has_complete_json_object(text))

    def test_markdown_opening_is_optional_and_closing_fence_is_not_required(self):
        for text in ('```json\n{"events":[]}', '```\n{"events":[]}\n```',
                     '  ```JSON\r\n{"events":[]}'):
            with self.subTest(text=text):
                self.assertTrue(_has_complete_json_object(text))

    def test_incomplete_malformed_and_nonobject_values_do_not_stop(self):
        for text in ('', '{', '{"events":', '{"events":[', '{"events":[]',
                     '{"events":[],}', '{events: []}', '"{escaped string}"',
                     '[]', '[{}]', 'true', '123', 'null', '```json',
                     '```python\n{"events":[]}', 'Explanation: {"events":[]}'):
            with self.subTest(text=text):
                self.assertFalse(_has_complete_json_object(text))

    def test_quoted_braces_and_escaped_quotes_are_decoded_as_strings(self):
        self.assertFalse(_has_complete_json_object('{"description":"}'))
        self.assertFalse(_has_complete_json_object('{"description":"} {", "events":['))
        self.assertTrue(_has_complete_json_object('{"description":"} {", "events":[]}'))
        self.assertTrue(_has_complete_json_object('{"description":"\\\"} {\\\"", "events":[]}'))

    def test_stopping_criterion_decodes_only_bounded_response_tokens(self):
        processor = MagicMock()
        processor.batch_decode.return_value = ['{"events":[]}']
        input_ids = MagicMock()
        input_ids.shape = (1, 150)
        criterion = _StopAfterJsonObject(processor, prompt_length=99, max_new_tokens=32)
        self.assertTrue(criterion(input_ids, scores=None))
        input_ids.__getitem__.assert_called_once_with((slice(None), slice(99, 131)))
        processor.batch_decode.assert_called_once_with(
            input_ids.__getitem__.return_value, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def test_no_response_tokens_do_not_decode_prompt(self):
        processor = MagicMock()
        criterion = _StopAfterJsonObject(processor, prompt_length=99, max_new_tokens=32)
        self.assertFalse(criterion(SimpleNamespace(shape=(1, 99)), scores=None))
        self.assertFalse(criterion(SimpleNamespace(shape=(2, 120)), scores=None))
        processor.batch_decode.assert_not_called()


class ConstrainedSchemaTests(unittest.TestCase):
    """Exercise the actual optional grammar parser without GPU or model weights."""

    def setUp(self):
        try:
            from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        except ImportError:
            self.skipTest("Install the model extra to test the actual JSON grammar")
        self.parser_type = JsonSchemaParser
        self.config = CharacterLevelParserConfig(max_consecutive_whitespaces=1,
                                                force_json_field_order=True)
        self.config.alphabet += "".join(chr(code) for code in range(0x0400, 0x0500))
        self.schemas = {phase: _build_output_schema(RULES, [10.25, 11.5], phase)
                        for phase in ("coarse", "fine")}

    def accepts(self, text, phase="coarse"):
        parser = self.parser_type(self.schemas[phase], config=self.config)
        for character in text:
            if character not in parser.get_allowed_characters():
                return False
            parser = parser.add_character(character)
        return parser.can_end()

    @staticmethod
    def event(**changes):
        result = {"rule_id": "demo_rule", "timecode_sec": 10.25,
                  "description": COARSE_DESCRIPTION}
        result.update(changes)
        return result

    def test_valid_empty_and_multiple_events_allow_completion(self):
        for value in ({"events": []}, {"events": [self.event()]},
                      {"events": [self.event()] * MAX_WINDOW_EVENTS}):
            with self.subTest(value=value):
                self.assertTrue(self.accepts(json.dumps(value, ensure_ascii=False)))

    def test_unknown_ids_unsupplied_times_and_long_description_are_blocked(self):
        for event in (self.event(rule_id="invented_rule"), self.event(timecode_sec=12.0),
                      self.event(description="x" * (DESCRIPTION_MAX_CHARS + 1))):
            with self.subTest(event=event):
                self.assertFalse(self.accepts(json.dumps({"events": [event]}, ensure_ascii=False)))

    def test_fine_accepts_one_short_description_and_blocks_long_or_multiple_events(self):
        event = self.event(description="Тестовое видимое действие.")
        self.assertTrue(self.accepts(json.dumps({"events": [event]}, ensure_ascii=False), "fine"))
        self.assertFalse(self.accepts(json.dumps({"events": [event, event]}, ensure_ascii=False), "fine"))
        event = self.event(description="x" * (DESCRIPTION_MAX_CHARS + 1))
        self.assertFalse(self.accepts(json.dumps({"events": [event]}, ensure_ascii=False), "fine"))

    def test_coarse_description_is_fixed_instead_of_generated_narrative(self):
        event = self.event(description="Произвольное описание кандидата")
        self.assertFalse(self.accepts(json.dumps({"events": [event]}, ensure_ascii=False)))

    def test_maximum_array_length_and_unknown_legal_field_are_blocked(self):
        self.assertFalse(self.accepts(json.dumps({"events": [self.event()] * (MAX_WINDOW_EVENTS + 1)})))
        event = self.event()
        event["law"] = "invented citation"
        self.assertFalse(self.accepts(json.dumps({"events": [event]})))

    def test_eos_is_unavailable_for_incomplete_or_malformed_object(self):
        for text in ('{"events":', '{"events":[{"rule_id":"demo_rule"}]}',
                     '{"events":[],}', '{"events":[],"verdict":"violation"}',
                     '{"events":[{"rule_id":"demo_rule","timecode_sec":10.25,"description":"x'):
            with self.subTest(text=text):
                self.assertFalse(self.accepts(text))


@unittest.skipUnless(os.getenv("CASHLENS_TEST_LOCAL_TOKENIZER") == "1",
                     "Set CASHLENS_TEST_LOCAL_TOKENIZER=1 for the local Qwen vocabulary regression")
class LocalTokenizerGrammarTests(unittest.TestCase):
    """CPU-only local vocabulary integration; never reads video or model weights."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

        directory = Path(__file__).resolve().parent.parent / "local" / "models" / "qwen3-vl-2b"
        if not directory.is_dir():
            raise unittest.SkipTest("Local Qwen tokenizer is unavailable")
        cls.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                                     trust_remote_code=False)
        cls.data = build_token_enforcer_tokenizer_data(cls.tokenizer)

    def test_empty_events_tokens_including_combined_array_and_object_are_allowed(self):
        import torch
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        for text in ('{"events":[]}', '{"events": []}', '{ "events" : [ ] }'):
            with self.subTest(text=text):
                config = CharacterLevelParserConfig(max_consecutive_whitespaces=1,
                                                    force_json_field_order=True)
                parser = JsonSchemaParser(_build_output_schema(RULES, [10.25]), config=config)
                prefix = build_transformers_prefix_allowed_tokens_fn(self.data, parser)
                parser.config = config
                sequence = [self.tokenizer.bos_token_id or 0]
                for token in self.tokenizer.encode(text, add_special_tokens=False):
                    allowed = prefix(0, torch.tensor(sequence))
                    self.assertIn(token, allowed, (text, self.tokenizer.decode([token])))
                    sequence.append(token)
                self.assertIn(self.tokenizer.eos_token_id, prefix(0, torch.tensor(sequence)))


if __name__ == "__main__":
    unittest.main()
