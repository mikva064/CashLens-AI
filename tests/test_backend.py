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
                                 _GpuDutyPacer, _build_prompt, _expand_wire_response,
                                 _has_complete_json_object)
from cashlens_ai.errors import InvalidModelOutputError, ModelUnavailableError, RuleValidationError


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
            set_per_process_memory_fraction=MagicMock(),
            synchronize=MagicMock(),
            memory_allocated=lambda _: 1024 * 1024 * 123,
            memory_reserved=lambda _: 1024 * 1024 * 456,
            max_memory_allocated=lambda _: 1024 * 1024 * 789,
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
        processor.batch_decode.return_value = ['{"e":[]}']
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

    def test_offline_generation_expands_compact_indices_to_absolute_times(self):
        torch, transformers, model, processor, batch = self.fake_dependencies()
        raw = '  {"e":[{"k":"demo_rule","f":1}]}  '
        processor.batch_decode.return_value = [raw]
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            result = self.backend.describe(self.frames, RULES, timeout_sec=7.5)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(json.loads(result), {"events": [{"rule_id": "demo_rule",
                         "timecode_sec": 11.5, "description": COARSE_DESCRIPTION}]})
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
        self.assertEqual(len(stopping), 2)
        self.assertIsInstance(stopping[0], _StopAfterJsonObject)
        self.assertIsInstance(stopping[1], _GpuDutyPacer)
        self.assertEqual(stopping[1].duty_cycle, 0.75)
        self.assertEqual(stopping[0].prompt_length, 99)
        self.assertEqual(stopping[0].max_new_tokens, self.backend.max_new_tokens)
        self.assertIs(
            model.generate.call_args.kwargs["prefix_allowed_tokens_fn"],
            transformers._lmfe_integration.build_transformers_prefix_allowed_tokens_fn.return_value,
        )
        schema_call = transformers._lmfe.JsonSchemaParser.call_args
        schema = schema_call.args[0]
        self.assertEqual(schema["properties"]["e"]["maxItems"], 3)
        event_properties = schema["properties"]["e"]["items"]["properties"]
        self.assertEqual(event_properties["k"]["enum"], ["demo_rule"])
        self.assertEqual(event_properties["f"]["enum"], [0, 1])
        self.assertEqual(set(event_properties), {"k", "f"})
        self.assertIn("Я", schema_call.kwargs["config"].alphabet)
        self.assertTrue(schema_call.kwargs["config"].force_json_field_order)
        self.assertIs(transformers._lmfe.JsonSchemaParser.return_value.config,
                      schema_call.kwargs["config"])
        chat = processor.apply_chat_template.call_args.args[0]
        all_text = " ".join(item.get("text", "") for item in chat[0]["content"])
        self.assertIn("10.250000 seconds", all_text)
        self.assertIn("11.500000 seconds", all_text)
        self.assertIn('"k": "demo_rule"', all_text)
        self.assertIn("Frame f=1", all_text)
        self.assertNotIn('"id": "demo_rule"', all_text)
        self.assertIn("Never identify people", all_text)
        self.assertIn("Do not invent laws", all_text)
        self.assertIn("First establish the visible scene", all_text)
        self.assertIn("complete sequence before and after", all_text)
        self.assertIn('evidence is missing, return {"e":[]}', all_text)
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
        self.assertEqual(torch.cuda.empty_cache.call_count, 4)  # load, 2 windows, close

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
            later_frames = [SimpleNamespace(timestamp_sec=value, image=object())
                            for value in (20.0, 20.5, 21.0)]
            self.backend.describe(later_frames, RULES, phase="fine",
                                  candidate_timecode_sec=20.5, candidate_radius_sec=0.25)
        transformers._lmfe_integration.build_token_enforcer_tokenizer_data.assert_called_once()
        self.assertEqual(transformers._lmfe.JsonSchemaParser.call_count, 2)
        schema = transformers._lmfe.JsonSchemaParser.call_args.args[0]
        times = schema["properties"]["e"]["items"]["properties"]["f"]
        self.assertEqual(times["enum"], [1])
        self.assertEqual(schema["properties"]["e"]["maxItems"], 3)
        self.assertEqual(set(schema["properties"]["e"]["items"]["properties"]), {"k", "b", "f", "a", "c"})

    def test_generation_diagnostics_contain_only_safe_metadata(self):
        torch, transformers, model, processor, _ = self.fake_dependencies()
        processor.batch_decode.return_value = ['{"events":[{"description":"PRIVATE OUTPUT"}']
        model.generate.return_value.__getitem__.return_value.shape = (1, 384)
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            with self.assertLogs("cashlens.model", level="INFO") as logs:
                with self.assertRaises(InvalidModelOutputError) as caught:
                    self.backend.describe(self.frames, RULES)
        self.assertEqual(caught.exception.reason_code, "response_truncated")
        diagnostic = " ".join(logs.output)
        self.assertIn("phase=coarse nframes=2 generated_tokens=384", diagnostic)
        self.assertIn("complete_json=False", diagnostic)
        self.assertIn("token_limit=True", diagnostic)
        self.assertIn("latency_sec=", diagnostic)
        self.assertNotIn("PRIVATE OUTPUT", diagnostic)
        self.assertNotIn(self.directory.name, diagnostic)
        self.assertIn("'allocated_mib': 123", diagnostic)
        self.assertIn("'reserved_mib': 456", diagnostic)
        self.assertIn("'peak_allocated_mib': 789", diagnostic)

    def test_complete_json_with_invalid_trailing_text_is_not_labeled_truncation(self):
        torch, transformers, model, processor, _ = self.fake_dependencies()
        processor.batch_decode.return_value = ['{"e":[]} PRIVATE_TRAILING_TEXT']
        model.generate.return_value.__getitem__.return_value.shape = (1, 384)
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            with self.assertRaises(InvalidModelOutputError) as caught:
                self.backend.describe(self.frames, RULES)
        self.assertEqual(caught.exception.reason_code, "invalid_compact_json")

    def test_prompts_use_bounded_phase_specific_descriptions(self):
        coarse = _build_prompt(RULES, "coarse")
        fine = _build_prompt(RULES, "fine")
        self.assertNotIn('{"e":[{"r":0,"f":0}]}', coarse)
        self.assertNotIn(COARSE_DESCRIPTION, coarse)
        self.assertIn("Do not emit a description", coarse)
        self.assertIn("at most three events", fine)
        self.assertIn("correct their category", fine)
        self.assertNotIn(COARSE_DESCRIPTION, fine)
        self.assertIn("b (before), f (decisive action), a (after), c (cash/service", fine)
        self.assertIn("Require b < f < a", fine)

    def test_fine_without_ordered_proof_context_abstains_before_gpu_loading(self):
        with patch.object(self.backend, "_load_model") as load:
            result = self.backend.describe(self.frames, RULES, phase="fine",
                                           candidate_timecode_sec=10.25, candidate_radius_sec=2)
        self.assertEqual(result, '{"events":[]}')
        load.assert_not_called()

    def test_memory_limit_is_applied_after_loaded_weights_and_cache_release(self):
        torch, transformers, model, *_ = self.fake_dependencies()
        events = []
        model.eval.side_effect = lambda: events.append("weights_loaded") or model
        torch.cuda.empty_cache.side_effect = lambda: events.append("cache_released")
        torch.cuda.set_per_process_memory_fraction.side_effect = lambda *args, **kwargs: events.append("limit_set")
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            self.backend.load()
        self.assertEqual(events, ["weights_loaded", "cache_released", "limit_set"])
        torch.cuda.set_per_process_memory_fraction.assert_called_once_with(0.8, device=0)

    def test_memory_limit_can_be_disabled_explicitly(self):
        self.backend = LocalQwenBackend(Path(self.directory.name), gpu_memory_fraction=None, gpu_duty_cycle=1)
        torch, transformers, *_ = self.fake_dependencies()
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            self.backend.load()
        torch.cuda.set_per_process_memory_fraction.assert_not_called()

    def test_releasing_unused_cache_keeps_loaded_weights(self):
        torch, transformers, model, *_ = self.fake_dependencies()
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            self.backend.load()
            count = torch.cuda.empty_cache.call_count
            self.backend.release_unused_memory()
        self.assertIs(self.backend._model, model)
        self.assertEqual(torch.cuda.empty_cache.call_count, count + 1)

    def test_inference_failure_releases_window_cache(self):
        torch, transformers, model, *_ = self.fake_dependencies()
        model.generate.side_effect = RuntimeError("synthetic inference failure")
        with patch.dict(sys.modules, self.dependency_modules(torch, transformers)), patch.dict(os.environ):
            with self.assertRaises(ModelUnavailableError):
                self.backend.describe(self.frames, RULES)
        self.assertEqual(torch.cuda.empty_cache.call_count, 2)  # load and failed window

    def test_invalid_resource_controls_are_rejected_before_loading(self):
        for name in ("gpu_duty_cycle", "gpu_memory_fraction"):
            for value in (0.0, 0.09, 1.01, float("nan"), float("inf"), True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    LocalQwenBackend(Path(self.directory.name), **{name: value})


class GpuPacingTests(unittest.TestCase):
    @staticmethod
    def fake_clock():
        timer = SimpleNamespace(value=0.0, pauses=[])

        def sleep(seconds):
            timer.pauses.append(seconds)
            timer.value += seconds

        return timer, lambda: timer.value, sleep

    def test_completed_steps_receive_proportional_idle_time_and_never_stop(self):
        timer, clock, sleep = self.fake_clock()
        cuda = SimpleNamespace(synchronize=MagicMock())
        pacer = _GpuDutyPacer(cuda, 0.75, clock=clock, sleep=sleep)
        for work in (0.3, 0.6, 0.15):
            timer.value += work
            self.assertFalse(pacer(None, None))
        self.assertAlmostEqual(pacer.total_sleep_sec, 0.35)
        self.assertAlmostEqual(1.05 / timer.value, 0.75)
        self.assertEqual(cuda.synchronize.call_count, 3)

    def test_long_prefill_idle_debt_is_repaid_in_bounded_pauses(self):
        timer, clock, sleep = self.fake_clock()
        pacer = _GpuDutyPacer(SimpleNamespace(synchronize=MagicMock()), 0.5,
                             clock=clock, sleep=sleep)
        timer.value = 10
        for _ in range(10):
            self.assertFalse(pacer(None, None))
        self.assertTrue(all(0 < pause <= 1 for pause in timer.pauses))
        self.assertAlmostEqual(pacer.total_sleep_sec, 10)
        self.assertAlmostEqual(10 / timer.value, 0.5)

    def test_full_duty_performs_no_sleep_or_extra_cuda_sync(self):
        timer, clock, sleep = self.fake_clock()
        cuda = SimpleNamespace(synchronize=MagicMock())
        pacer = _GpuDutyPacer(cuda, 1, clock=clock, sleep=sleep)
        timer.value = 20
        self.assertFalse(pacer(None, None))
        self.assertEqual(timer.pauses, [])
        cuda.synchronize.assert_not_called()


class CompactWireTests(unittest.TestCase):
    def test_semantic_keys_restore_trusted_id_and_exact_global_timestamp(self):
        rules = RULES + [{**RULES[0], "id": "another_reviewed_rule"}]
        timestamps = [123.1234567890123, 124.987654321]
        result = json.loads(_expand_wire_response('{"e":[{"k":"another_reviewed_rule","f":0}]}', rules, timestamps))
        self.assertEqual(result, {"events": [{"rule_id": "another_reviewed_rule",
                         "timecode_sec": timestamps[0], "description": COARSE_DESCRIPTION}]})
        self.assertEqual(_expand_wire_response('{"e":[]}', rules, timestamps), '{"events":[]}')

    def test_fine_proof_expands_existing_pipeline_schema_with_trusted_template(self):
        from cashlens_ai.pipeline import parse_model_response
        timestamps = [120.123456789, 121.987654321, 125.123456789]
        rules = [{**RULES[0], "description_template": "Тестовое действие."}]
        text = _expand_wire_response('{"e":[{"k":"demo_rule","b":0,"f":1,"a":2,"c":0}]}',
                                     rules, timestamps, "fine", candidate_timecodes_sec=[timestamps[1]],
                                     candidate_radius_sec=0.5)
        result = parse_model_response(text, allowed_ids={"demo_rule"}, first_sec=timestamps[0],
            last_sec=timestamps[-1], require_proof=True, frame_timestamps=timestamps,
            candidate_timecodes_sec=[timestamps[1]], candidate_radius_sec=0.5)
        self.assertEqual(result[0].rule_id, "demo_rule")
        self.assertEqual(result[0].timecode_sec, timestamps[1])
        self.assertEqual(result[0].description, "Тестовое действие.")
        self.assertEqual(result[0].evidence_timestamps_sec, timestamps)

    def test_permutation_preserves_semantic_rule_and_identical_prompt(self):
        other = {**RULES[0], "id": "another_reviewed_rule"}
        orders = (RULES + [other], [other] + RULES)
        for rules in orders:
            result = json.loads(_expand_wire_response('{"e":[{"k":"another_reviewed_rule","f":0}]}', rules, [12.]))
            self.assertEqual(result["events"][0]["rule_id"], other["id"])
        self.assertEqual(_build_prompt(orders[0], "coarse"), _build_prompt(orders[1], "coarse"))

    def test_invalid_indices_keys_and_extra_legal_fields_fail_closed(self):
        for event in ({"k": True, "f": 0}, {"k": "unknown", "f": 0}, {"k": "demo_rule", "f": False},
                      {"k": "demo_rule", "f": 1.0}, {"k": "demo_rule", "f": -1},
                      {"k": "demo_rule", "f": 2}, {"k": "demo_rule"},
                      {"k": "demo_rule", "f": 0, "law": "invented"}):
            with self.subTest(event=event), self.assertRaises(InvalidModelOutputError):
                _expand_wire_response(json.dumps({"e": [event]}), RULES, [12., 13.])

    def test_invalid_json_duplicate_keys_and_nonfinite_constants_are_rejected(self):
        for text in ('{"e":', '{"e":[],"e":[]}', '{"e":[{"k":"demo_rule","k":"demo_rule","f":0}]}',
                     '{"e":[{"k":"demo_rule","f":NaN}]}', '{"e":[]} trailing',
                     '{"events":[]}', '{"e":null}', '{"e":{}}', '[]'):
            with self.subTest(text=text), self.assertRaises(InvalidModelOutputError):
                _expand_wire_response(text, RULES, [12.])

    def test_fine_rejects_missing_reversed_unrelated_or_unfocused_proof(self):
        valid = {"k": "demo_rule", "b": 0, "f": 1, "a": 2, "c": 0}
        invalid = [{key: value for key, value in valid.items() if key != missing} for missing in ("b", "a", "c", "k")]
        invalid += [{**valid, "b": 1}, {**valid, "a": 0}, {**valid, "c": 3}, {**valid, "b": False},
                    {**valid, "a": 99}, {**valid, "f": 2, "a": 3}, {**valid, "d": "untrusted prose"}]
        for event in invalid:
            with self.subTest(event=event), self.assertRaises(InvalidModelOutputError):
                _expand_wire_response(json.dumps({"e": [event]}), RULES, [12.,13.,14.,15.], "fine",
                                      candidate_timecodes_sec=[13.], candidate_radius_sec=0.25)
        with self.assertRaises(InvalidModelOutputError):
            _expand_wire_response(json.dumps({"e": [valid]}), RULES, [12.,13.,13.], "fine",
                                  candidate_timecodes_sec=[13.], candidate_radius_sec=0.25)

    def test_fine_batch_can_correct_type_and_use_any_candidate_focus(self):
        rules = RULES + [{**RULES[0], "id": "corrected_category"}]
        events = [{"k":"demo_rule","b":0,"f":1,"a":2,"c":1},
                  {"k":"corrected_category","b":2,"f":3,"a":4,"c":3}]
        result = json.loads(_expand_wire_response(json.dumps({"e":events}), rules, [10.,11.,12.,13.,14.],
            "fine", candidate_timecodes_sec=[11.,13.], candidate_radius_sec=0.2))
        self.assertEqual([event["rule_id"] for event in result["events"]], ["demo_rule","corrected_category"])
        with self.assertRaises(InvalidModelOutputError):
            _expand_wire_response(json.dumps({"e":[events[0]]*2}), rules, [10.,11.,12.,13.,14.],
                "fine", candidate_timecodes_sec=[11.], candidate_radius_sec=0.2)


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
            # Character grammar needs no Torch. Avoid loading CUDA DLLs beside
            # an active model service merely to exercise JSON transitions.
            with patch.dict(sys.modules, {"torch": None}):
                from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        except ImportError:
            self.skipTest("Install the model extra to test the actual JSON grammar")
        self.parser_type = JsonSchemaParser
        self.config = CharacterLevelParserConfig(max_consecutive_whitespaces=1,
                                                force_json_field_order=True)
        self.config.alphabet += "".join(chr(code) for code in range(0x0400, 0x0500))
        self.schemas = {"coarse": _build_output_schema(RULES, [10.25, 11.5]),
                        "fine": _build_output_schema(RULES, [10.25, 11.5, 13.0], "fine",
                                                     candidate_timecode_sec=11.5,
                                                     candidate_radius_sec=0.25)}

    def accepts(self, text, phase="coarse"):
        parser = self.parser_type(self.schemas[phase], config=self.config)
        for character in text:
            if character not in parser.get_allowed_characters():
                return False
            parser = parser.add_character(character)
        return parser.can_end()

    @staticmethod
    def event(**changes):
        result = {"k": "demo_rule", "f": 0}
        result.update(changes)
        return result

    def test_valid_empty_and_multiple_events_allow_completion(self):
        for value in ({"e": []}, {"e": [self.event()]},
                      {"e": [self.event()] * MAX_WINDOW_EVENTS}):
            with self.subTest(value=value):
                self.assertTrue(self.accepts(json.dumps(value, ensure_ascii=False)))

    def test_unknown_rule_and_frame_indices_are_blocked(self):
        for event in (self.event(k="unknown"), self.event(f=2), self.event(k=True), self.event(f=True)):
            with self.subTest(event=event):
                self.assertFalse(self.accepts(json.dumps({"e": [event]}, ensure_ascii=False)))

    def test_fine_grammar_is_bounded_structured_proof_without_generated_prose(self):
        event = {"k":"demo_rule", "b":0,"f":1,"a":2,"c":0}
        self.assertTrue(self.accepts(json.dumps({"e":[event]}), "fine"))
        self.assertFalse(self.accepts(json.dumps({"e":[event]*4}), "fine"))
        self.assertFalse(self.accepts(json.dumps({"e":[{**event,"d":"not allowed"}]}), "fine"))

    def test_coarse_description_cannot_be_generated(self):
        event = self.event(description="Произвольное описание кандидата")
        self.assertFalse(self.accepts(json.dumps({"e": [event]}, ensure_ascii=False)))

    def test_maximum_array_length_and_unknown_legal_field_are_blocked(self):
        self.assertFalse(self.accepts(json.dumps({"e": [self.event()] * (MAX_WINDOW_EVENTS + 1)})))
        event = self.event()
        event["law"] = "invented citation"
        self.assertFalse(self.accepts(json.dumps({"e": [event]})))

    def test_eos_is_unavailable_for_incomplete_or_malformed_object(self):
        for text in ('{"e":', '{"e":[{"r":0}]}', '{"e":[],}',
                     '{"e":[],"verdict":"violation"}', '{"e":[{"r":0,"f":'):
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

        for text in ('{"e":[]}', '{"e": []}', '{ "e" : [ ] }'):
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

    def test_compact_candidates_use_fewer_output_tokens_without_changing_frames(self):
        wire = '{"e":[{"k":"demo_rule","f":0},{"k":"demo_rule","f":1}]}'
        expanded = _expand_wire_response(wire, RULES, [120.123456789, 121.987654321])
        compact_count = len(self.tokenizer.encode(wire, add_special_tokens=False))
        expanded_count = len(self.tokenizer.encode(expanded, add_special_tokens=False))
        self.assertLess(compact_count, expanded_count / 2)


if __name__ == "__main__":
    unittest.main()
