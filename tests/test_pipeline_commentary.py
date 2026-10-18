"""Episode comments must use their own proof frames, without changing detection."""
import logging
from dataclasses import replace

import pytest

from cashlens_ai.contracts import EventCommentary, Rule, RuleSource
from cashlens_ai.pipeline import AnalyzerConfig, VideoAnalyzer
from test_pipeline import RecordingBackend, encode_fixture, event, proof_event, response


@pytest.fixture
def setup_video(tmp_path):
    video = tmp_path / "synthetic.mov"
    encode_fixture(video)
    rule = Rule(id="synthetic_event", title="Synthetic category criterion",
                observable_condition="A synthetic movement is visible in ordered frames.", verified=True,
                source=RuleSource(document="Synthetic source", section="1", url="https://example.invalid/rule"))
    config = AnalyzerConfig(sample_fps=2, window_sec=3, overlap_sec=.5, max_refine_frames=12,
                            max_edge=64, refine_max_edge=80, merge_sec=.1)
    return video, rule, config


class CommentBackend(RecordingBackend):
    def __init__(self, points=(.6,), caption=None):
        self.caption_calls = []
        self.first = True
        self.points = points
        self.caption = caption or (lambda frames, timeout: {
            "status": "generated", "text": f"Тестовое движение на кадре {frames[1].timestamp_sec:.3f}."})
        super().__init__(self.detect)

    def detect(self, phase, frames, rules):
        if phase == "coarse":
            if not self.first:
                return response()
            self.first = False
            return response(*(event(point) for point in self.points))
        return response(*(proof_event(frames, action_sec=point) for point in self.points))

    def comment_event(self, frames, *, timeout_sec):
        self.caption_calls.append({"timestamps": [frame.timestamp_sec for frame in frames],
                                   "timeout_sec": timeout_sec})
        return self.caption(frames, timeout_sec)


def analyze(setup_video, tmp_path, backend, *, config=None, clock=None, directory="result"):
    video, rule, default = setup_video
    options = {"clock": clock} if clock is not None else {}
    return VideoAnalyzer(backend, [rule], config=config or default, **options).analyze(video, tmp_path / directory)


def test_each_comment_uses_its_own_exact_proof_frames_and_not_the_category_template(setup_video, tmp_path):
    backend = CommentBackend(points=(.6, 1.6))
    result = analyze(setup_video, tmp_path, backend)
    assert result.verdict == "violation"
    assert len(backend.caption_calls) == len(result.violations) == 2
    assert len({item.description for item in result.violations}) == 2
    for item, call in zip(result.violations, backend.caption_calls):
        assert item.commentary_status == "generated"
        assert item.category_description == setup_video[1].title
        assert item.description != item.category_description
        assert call["timestamps"] == [frame.timestamp_sec for frame in item.evidence_frames]
        assert f"{item.timecode_sec:.3f}" in item.description
        assert item.source == setup_video[1].source


@pytest.mark.parametrize("caption", [
    lambda frames, timeout: {"status": "uncertain", "text": ""},
    lambda frames, timeout: {"status": "generated", "text": ""},
    lambda frames, timeout: {"status": "unknown", "text": "PRIVATE_SCENE_TEXT"},
    lambda frames, timeout: {"status": "generated", "text": {"private": "PRIVATE_SCENE_TEXT"}},
])
def test_uncertain_or_malformed_caption_keeps_event_but_never_uses_category_as_scene(setup_video, tmp_path, caption):
    result = analyze(setup_video, tmp_path, CommentBackend(caption=caption))
    item = result.violations[0]
    assert result.verdict == "violation"
    assert item.commentary_status in {"uncertain", "unavailable"}
    assert item.description != setup_video[1].title
    assert "PRIVATE_SCENE_TEXT" not in item.description
    assert "00:00.600" in item.description


def test_caption_error_does_not_change_detection_or_log_private_error_text(setup_video, tmp_path, caplog):
    def failed(frames, timeout):
        raise ValueError("PRIVATE_SCENE_TEXT")
    caplog.set_level(logging.INFO, logger="cashlens.pipeline")
    result = analyze(setup_video, tmp_path, CommentBackend(caption=failed))
    assert result.verdict == "violation"
    assert result.violations[0].commentary_status == "unavailable"
    assert "error_type=ValueError" in caplog.text
    assert "PRIVATE_SCENE_TEXT" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_captioning_does_not_change_detector_calls_category_time_source_or_proof(setup_video, tmp_path):
    enabled = CommentBackend(points=(.6, 1.6))
    disabled = CommentBackend(points=(.6, 1.6))
    first = analyze(setup_video, tmp_path, enabled, directory="enabled", clock=lambda: 0.)
    second = analyze(setup_video, tmp_path, disabled, directory="disabled",
                     config=replace(setup_video[2], commentary_budget_sec=0), clock=lambda: 0.)
    assert enabled.calls == disabled.calls
    assert disabled.caption_calls == []
    signature = lambda result: [(e.rule_id, e.timecode_sec, e.source,
                                 [f.timestamp_sec for f in e.evidence_frames]) for e in result.violations]
    assert signature(first) == signature(second)
    assert first.verdict == second.verdict


def test_total_caption_budget_is_shared_between_events(setup_video, tmp_path):
    now = [0.0]
    def caption(frames, timeout):
        now[0] += min(8., timeout)
        return {"status": "generated", "text": "Тестовая фигура перемещается между кадрами."}
    backend = CommentBackend(points=(.6, 1., 1.6), caption=caption)
    result = analyze(setup_video, tmp_path, backend, clock=lambda: now[0],
                     config=replace(setup_video[2], commentary_budget_sec=20., commentary_timeout_sec=10.))
    assert [call["timeout_sec"] for call in backend.caption_calls] == [10., 10., 4.]
    assert now[0] == 20.
    assert len(result.violations) == 3


def test_caption_is_skipped_near_the_existing_analysis_deadline(setup_video, tmp_path):
    now = [0.0]
    backend = CommentBackend()
    detect = backend.callback
    def near_deadline(phase, frames, rules):
        result = detect(phase, frames, rules)
        if phase == "fine":
            now[0] = 233.
        return result
    backend.callback = near_deadline
    result = analyze(setup_video, tmp_path, backend, clock=lambda: now[0])
    assert result.verdict == "violation"
    assert backend.caption_calls == []
    assert result.violations[0].commentary_status == "unavailable"


def test_no_caption_call_for_a_negative_detection(setup_video, tmp_path):
    backend = CommentBackend(points=())
    result = analyze(setup_video, tmp_path, backend)
    assert result.verdict == "no_violation"
    assert backend.caption_calls == []


@pytest.mark.parametrize("field,value", [("commentary_budget_sec", -1), ("commentary_budget_sec", float("nan")),
                                        ("commentary_timeout_sec", 0), ("commentary_timeout_sec", float("inf"))])
def test_invalid_comment_budget_is_rejected(field, value):
    with pytest.raises(ValueError):
        AnalyzerConfig(**{field: value})


def test_commentary_contract_rejects_uncertain_assertions():
    with pytest.raises(ValueError):
        EventCommentary(status="uncertain", text="An unsupported scene assertion")
