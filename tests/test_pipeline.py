"""Integration checks with a generated MOV and a deterministic model stand-in."""

from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import av
from PIL import Image

from cashlens_ai.contracts import Rule, RuleSource
from cashlens_ai.errors import (
    AnalysisTimeoutError,
    InvalidModelOutputError,
    RuleValidationError,
)
from cashlens_ai.pipeline import AnalyzerConfig, VideoAnalyzer
from cashlens_ai.video import iter_sampled_frames


def encode_fixture(path: Path) -> None:
    # Source PTS start at five seconds. Variable presentation intervals make
    # accidental index/fps timestamps distinguishable from source timestamps.
    points = (50, 52, 54, 56, 58, 60, 62, 64, 66, 69, 70, 72, 74, 76, 78)
    with av.open(str(path), mode="w", format="mov") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width = 80
        stream.height = 40
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 10)
        stream.codec_context.gop_size = 2
        for index, pts in enumerate(points):
            frame = av.VideoFrame.from_image(Image.new("RGB", (80, 40), (index * 15, 20, 30)))
            frame.pts = pts
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def response(*events):
    return json.dumps({"events": list(events)})


def event(timestamp, *, rule_id="synthetic_event"):
    return {"rule_id": rule_id, "timecode_sec": timestamp, "description": "Synthetic visible event"}


class RecordingBackend:
    def __init__(self, callback=None):
        self.calls = []
        self.callback = callback or (lambda phase, frames, rules: response())

    def describe(self, frames, rules, *, phase="coarse", timeout_sec=None):
        self.calls.append({
            "phase": phase,
            "timestamps": [frame.timestamp_sec for frame in frames],
            "rules": rules,
            "timeout_sec": timeout_sec,
        })
        return self.callback(phase, frames, rules)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_temp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.fixture_temp.name) / "synthetic.mov"
        encode_fixture(cls.video)
        # This is an explicit synthetic fixture, not an asserted banking norm.
        cls.rule = Rule(
            id="synthetic_event",
            title="Synthetic event for tests",
            observable_condition="The synthetic action is visible in the test frames.",
            source=RuleSource(
                document="Synthetic test policy (not a legal source)",
                section="Test section 1",
                url="https://example.invalid/synthetic-test-policy",
            ),
            verified=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.fixture_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = AnalyzerConfig(
            sample_fps=2,
            window_sec=1.5,
            overlap_sec=0.5,
            refine_radius_sec=0.4,
            refine_fps=5,
            max_edge=64,
            time_budget_sec=30,
        )

    def tearDown(self):
        self.temp.cleanup()

    def analyzer(self, backend, **kwargs):
        return VideoAnalyzer(backend, [self.rule], config=self.config, **kwargs)

    def test_clean_verdict_requires_entire_coarse_scan_including_tail(self):
        output = self.root / "job"

        def no_event(phase, frames, rules):
            self.assertFalse((output / "result.json").exists())
            return response()

        backend = RecordingBackend(no_event)
        result = self.analyzer(backend).analyze(self.video, output)
        self.assertEqual(result.verdict, "no_violation")
        self.assertEqual(result.violations, [])
        self.assertEqual([call["phase"] for call in backend.calls], ["coarse"] * 3)
        observed = {round(ts, 6) for call in backend.calls for ts in call["timestamps"]}
        expected = {round(frame.timestamp_sec, 6) for frame in iter_sampled_frames(self.video, fps=2)}
        self.assertEqual(observed, expected)
        self.assertEqual([round(ts, 6) for ts in backend.calls[-1]["timestamps"]], [2.0, 2.6])
        self.assertTrue(all(call["timeout_sec"] > 0 for call in backend.calls))
        saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["verdict"], "no_violation")

    def test_coarse_then_fine_emits_evidence_at_actual_source_times(self):
        coarse_calls = 0

        def detect(phase, frames, rules):
            nonlocal coarse_calls
            if phase == "coarse":
                coarse_calls += 1
                return response(event(0.6)) if coarse_calls == 1 else response()
            self.assertEqual(phase, "fine")
            return response(event(0.73))

        backend = RecordingBackend(detect)
        output = self.root / "job"
        result = self.analyzer(backend).analyze(self.video, output)
        self.assertEqual(result.verdict, "violation")
        self.assertEqual(len(result.violations), 1)
        violation = result.violations[0]
        self.assertAlmostEqual(violation.timecode_sec, 0.8, places=6)
        self.assertEqual(violation.rule_id, self.rule.id)
        self.assertEqual(violation.source, self.rule.source)
        fine_calls = [call for call in backend.calls if call["phase"] == "fine"]
        self.assertEqual(len(fine_calls), 1)
        self.assertEqual([round(ts, 6) for ts in fine_calls[0]["timestamps"]],
                         [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        self.assertEqual([rule["id"] for rule in fine_calls[0]["rules"]], [self.rule.id])
        self.assertEqual([round(frame.timestamp_sec, 6) for frame in violation.evidence_frames], [0.2, 0.8, 1.2])
        for frame in violation.evidence_frames:
            self.assertEqual(frame.image_path.parent, output.resolve())
            self.assertTrue(frame.image_path.is_file())
            with Image.open(frame.image_path) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (80, 40))

    def test_fine_can_disconfirm_a_coarse_candidate(self):
        first_coarse = True

        def detect(phase, frames, rules):
            nonlocal first_coarse
            if phase == "coarse" and first_coarse:
                first_coarse = False
                return response(event(0.6))
            return response()

        backend = RecordingBackend(detect)
        result = self.analyzer(backend).analyze(self.video, self.root / "job")
        self.assertEqual(result.verdict, "no_violation")
        self.assertEqual(result.violations, [])
        self.assertEqual(len([call for call in backend.calls if call["phase"] == "fine"]), 1)

    def test_fine_retains_coarse_context_for_disconfirmation_under_frame_cap(self):
        source_context = []
        coarse_calls = 0

        def detect(phase, frames, rules):
            nonlocal coarse_calls
            timestamps = [round(frame.timestamp_sec, 6) for frame in frames]
            if phase == "coarse":
                coarse_calls += 1
                if coarse_calls == 2:
                    source_context.extend((timestamps[0], timestamps[-1]))
                    return response(event(1.6))
                return response()
            # The original beginning/end disprove the apparent middle action.
            # A narrow +/- radius or naive downsampling loses this evidence.
            if all(timestamp in timestamps for timestamp in source_context):
                return response()
            return response(event(1.6))

        config = replace(self.config, refine_radius_sec=0.3,
                         refine_fps=20, max_refine_frames=4)
        backend = RecordingBackend(detect)
        result = VideoAnalyzer(backend, [self.rule], config=config).analyze(
            self.video, self.root / "job")
        self.assertEqual(result.verdict, "no_violation")
        self.assertEqual(result.violations, [])
        fine_calls = [call for call in backend.calls if call["phase"] == "fine"]
        self.assertEqual(len(fine_calls), 1)
        self.assertEqual(source_context, [1.0, 2.0])
        fine_times = [round(ts, 6) for ts in fine_calls[0]["timestamps"]]
        self.assertEqual(fine_times, [0.8, 1.0, 2.0, 2.2])
        self.assertEqual(len(fine_times), config.max_refine_frames)

    def test_unknown_rule_out_of_range_or_malformed_output_fails(self):
        bad_outputs = (
            response(event(0.6, rule_id="unknown_event")),
            response(event(5.6)),  # Unnormalized source time, outside this window.
            response(event(-0.1)),
            "not json",
            json.dumps({"events": [{"rule_id": self.rule.id, "timecode_sec": 0.6}]}),
            json.dumps({"events": [], "extra": "unexpected"}),
        )
        for index, bad in enumerate(bad_outputs):
            with self.subTest(output=bad):
                output = self.root / f"job_{index}"
                backend = RecordingBackend(lambda phase, frames, rules, raw=bad: raw)
                with self.assertRaises(InvalidModelOutputError):
                    self.analyzer(backend).analyze(self.video, output)
                self.assertFalse((output / "result.json").exists())

    def test_fine_output_must_also_match_allowed_rule_and_window(self):
        for index, bad in enumerate((response(event(0.6, rule_id="unknown_event")), response(event(2.5)))):
            with self.subTest(output=bad):
                first_coarse = True

                def detect(phase, frames, rules, raw=bad):
                    nonlocal first_coarse
                    if phase == "coarse":
                        if first_coarse:
                            first_coarse = False
                            return response(event(0.6))
                        return response()
                    return raw

                output = self.root / f"job_{index}"
                with self.assertRaises(InvalidModelOutputError):
                    self.analyzer(RecordingBackend(detect)).analyze(self.video, output)
                self.assertFalse((output / "result.json").exists())

    def test_exhausted_budget_never_returns_clean_verdict(self):
        now = [0.0]

        def slow(phase, frames, rules):
            now[0] = 31.0
            return response()

        backend = RecordingBackend(slow)
        output = self.root / "job"
        with self.assertRaises(AnalysisTimeoutError):
            self.analyzer(backend, clock=lambda: now[0]).analyze(self.video, output)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["timeout_sec"], 30.0)
        self.assertFalse((output / "result.json").exists())

    def test_existing_output_directory_refuses_evidence_mixing(self):
        output = self.root / "job"
        output.mkdir()
        sentinel = output / "earlier-evidence.jpg"
        sentinel.write_bytes(b"previous job")
        backend = RecordingBackend()
        with self.assertRaises(FileExistsError):
            self.analyzer(backend).analyze(self.video, output)
        self.assertEqual(backend.calls, [])
        self.assertEqual(sentinel.read_bytes(), b"previous job")
        self.assertFalse((output / "result.json").exists())

    def test_invalid_rule_registry_fails_before_model_calls(self):
        backend = RecordingBackend()
        for rules in ([], [self.rule.model_copy(update={"verified": False})], [self.rule, self.rule]):
            with self.subTest(rules=rules), self.assertRaises(RuleValidationError):
                VideoAnalyzer(backend, rules, config=self.config)
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
