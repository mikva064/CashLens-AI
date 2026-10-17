"""Backend boundaries without weights, network access or a CUDA installation."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cashlens_ai.backend import LocalQwenBackend
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
        return torch, transformers, model, processor, batch

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
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.dict(os.environ):
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
        chat = processor.apply_chat_template.call_args.args[0]
        all_text = " ".join(item.get("text", "") for item in chat[0]["content"])
        self.assertIn("10.250000 seconds", all_text)
        self.assertIn("11.500000 seconds", all_text)
        self.assertIn('"id": "demo_rule"', all_text)
        self.assertIn("Never identify people", all_text)
        self.assertIn("Do not invent laws", all_text)
        self.assertEqual(processor.call_args.kwargs["images"], [frame.image for frame in self.frames])

    def test_cuda_absence_raises_error_without_cpu_or_empty_result(self):
        torch, transformers, *_ = self.fake_dependencies(cuda=False)
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.dict(os.environ):
            with self.assertRaisesRegex(ModelUnavailableError, "CUDA GPU"):
                self.backend.describe(self.frames, RULES)
        transformers.Qwen3VLForConditionalGeneration.from_pretrained.assert_not_called()

    def test_fp16_and_unquantized_option(self):
        self.backend = LocalQwenBackend(Path(self.directory.name), quantize_4bit=False)
        torch, transformers, *_ = self.fake_dependencies(bf16=False)
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.dict(os.environ):
            self.backend.describe(self.frames, RULES)
        options = transformers.Qwen3VLForConditionalGeneration.from_pretrained.call_args.kwargs
        self.assertEqual(options["dtype"], "fp16")
        self.assertNotIn("quantization_config", options)

    def test_load_error_is_not_reported_as_no_violation(self):
        torch, transformers, *_ = self.fake_dependencies()
        transformers.Qwen3VLForConditionalGeneration.from_pretrained.side_effect = RuntimeError("private path")
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.dict(os.environ):
            with self.assertRaises(ModelUnavailableError) as error:
                self.backend.describe(self.frames, RULES)
        self.assertNotIn("private path", str(error.exception))

    def test_empty_generation_is_an_error(self):
        torch, transformers, _, processor, _ = self.fake_dependencies()
        processor.batch_decode.return_value = [""]
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.dict(os.environ):
            with self.assertRaisesRegex(ModelUnavailableError, "no usable text"):
                self.backend.describe(self.frames, RULES)

    def test_model_is_reused_and_close_releases_references(self):
        torch, transformers, *_ = self.fake_dependencies()
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.dict(os.environ):
            self.backend.describe(self.frames, RULES)
            self.backend.describe(self.frames, RULES)
            self.backend.close()
        self.assertEqual(transformers.Qwen3VLForConditionalGeneration.from_pretrained.call_count, 1)
        self.assertIsNone(self.backend._model)
        self.assertIsNone(self.backend._processor)
        torch.cuda.empty_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
