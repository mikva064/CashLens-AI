"""Coarse-to-fine local analysis with validated rule IDs and source evidence."""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from pydantic import Field, ValidationError

from .contracts import AnalysisResult, EvidenceFrame, Rule, StrictModel, Violation
from .errors import AnalysisTimeoutError, InvalidModelOutputError, RuleValidationError, VideoReadError
from .video import SampledFrame, get_frame_at, iter_sampled_frames, probe_video


class ModelEvent(StrictModel):
    rule_id: str
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1, max_length=1600)


class ModelResponse(StrictModel):
    events: list[ModelEvent] = Field(max_length=32)


class WindowBackend(Protocol):
    def describe(self, frames: Sequence[SampledFrame], rules: list[dict], *,
                 phase: str = "coarse", timeout_sec: float | None = None) -> str: ...


@dataclass(frozen=True)
class AnalyzerConfig:
    sample_fps: float = 1.0
    window_sec: float = 8.0
    overlap_sec: float = 2.0
    refine_radius_sec: float = 2.0
    refine_fps: float = 3.0
    max_refine_frames: int = 12
    max_edge: int = 448
    refine_max_edge: int = 896
    merge_sec: float = 2.0
    time_budget_sec: float = 240.0

    def __post_init__(self):
        for name in ("sample_fps", "window_sec", "refine_radius_sec", "refine_fps",
                     "time_budget_sec"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.overlap_sec) or not 0 <= self.overlap_sec < self.window_sec:
            raise ValueError("overlap_sec must be between zero and window_sec")
        if not math.isfinite(self.merge_sec) or self.merge_sec < 0:
            raise ValueError("merge_sec must be finite and nonnegative")
        if self.max_edge < 32 or self.refine_max_edge < 32 or self.max_refine_frames < 2:
            raise ValueError("Both max_edge values >= 32 and max_refine_frames >= 2 are required")


def parse_model_response(text: str, *, allowed_ids: set[str],
                         first_sec: float, last_sec: float) -> list[ModelEvent]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
    try:
        data, end = json.JSONDecoder().raw_decode(text)
        # Some local generations add one unmatched presentation quote after a
        # complete object. Accept that exact formatting defect, never prose,
        # a second result or an incomplete object. Schema/rule checks still apply.
        if not isinstance(data, dict) or text[end:].strip() not in {"", '"', "'"}:
            raise ValueError("Unexpected text after event JSON")
        result = ModelResponse.model_validate(data)
    except (ValueError, ValidationError) as exc:
        raise InvalidModelOutputError("Model response is not valid event JSON") from exc
    for event in result.events:
        if event.rule_id not in allowed_ids:
            raise InvalidModelOutputError("Model returned an unknown rule ID")
        if not first_sec - 0.001 <= event.timecode_sec <= last_sec + 0.001:
            raise InvalidModelOutputError("Model timestamp is outside the supplied frames")
    return result.events


def merge_events(events: list[ModelEvent], threshold_sec: float) -> list[ModelEvent]:
    """Collapse overlapping-window detections, preserving the earliest anchor.

    This is a tunable heuristic: repeated same-type acts within threshold_sec can
    be merged. Keep the threshold small and measure it on local annotations.
    """
    output: list[ModelEvent] = []
    for event in sorted(events, key=lambda e: (e.timecode_sec, e.rule_id)):
        duplicate = any(e.rule_id == event.rule_id and
                        abs(e.timecode_sec - event.timecode_sec) <= threshold_sec
                        for e in output)
        if not duplicate:
            output.append(event)
    return output


class VideoAnalyzer:
    def __init__(self, backend: WindowBackend, rules: list[Rule], *,
                 config: AnalyzerConfig | None = None,
                 clock: Callable[[], float] = time.monotonic):
        if not rules or any(not r.verified for r in rules):
            raise RuleValidationError("A nonempty, verified rule registry is required")
        if len({r.id for r in rules}) != len(rules):
            raise RuleValidationError("Duplicate rule IDs")
        self.backend = backend
        self.rules = {r.id: r for r in rules}
        self.config = config or AnalyzerConfig()
        self.clock = clock

    def analyze(self, video_path: str | Path, output_dir: str | Path) -> AnalysisResult:
        config = self.config
        started = self.clock()

        def remaining():
            value = config.time_budget_sec - (self.clock() - started)
            if value <= 0:
                raise AnalysisTimeoutError("Analysis budget exceeded; result is incomplete")
            return value

        info = probe_video(Path(video_path))
        # Each job must own a fresh output directory; never mix evidence across jobs.
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)

        def analyze_window(frames, rules, phase):
            raw = self.backend.describe(
                frames, [r.model_dump(mode="json") for r in rules],
                phase=phase, timeout_sec=remaining(),
            )
            remaining()
            return parse_model_response(
                raw, allowed_ids={r.id for r in rules},
                first_sec=frames[0].timestamp_sec, last_sec=frames[-1].timestamp_sec,
            )

        frames_per_window = max(2, round(config.window_sec * config.sample_fps))
        overlap = min(frames_per_window - 1, round(config.overlap_sec * config.sample_fps))
        buffer: list[SampledFrame] = []
        fresh = 0
        windows_analyzed = 0
        candidates: list[ModelEvent] = []
        candidate_context: dict[tuple[str, float], tuple[float, float]] = {}

        def collect_candidates(frames):
            for event in analyze_window(frames, list(self.rules.values()), "coarse"):
                candidates.append(event)
                # merge_events preserves the earliest anchor. Keep the context
                # of its first originating window without changing ModelEvent.
                candidate_context.setdefault(
                    (event.rule_id, event.timecode_sec),
                    (frames[0].timestamp_sec, frames[-1].timestamp_sec),
                )

        for frame in iter_sampled_frames(info.path, fps=config.sample_fps, max_edge=config.max_edge):
            remaining()
            buffer.append(frame)
            fresh += 1
            if len(buffer) == frames_per_window:
                collect_candidates(buffer)
                windows_analyzed += 1
                buffer = buffer[-overlap:] if overlap else []
                fresh = 0
        if fresh:
            collect_candidates(buffer)
            windows_analyzed += 1
        if not windows_analyzed:
            raise VideoReadError("No video windows were analyzed")
        candidates = merge_events(candidates, config.merge_sec)

        confirmed: list[ModelEvent] = []
        for event in candidates:
            remaining()
            first_sec, last_sec = candidate_context[(event.rule_id, event.timecode_sec)]
            # Verification must retain the before/after context that produced
            # the candidate, including service/transaction boundaries.
            start = max(0.0, first_sec - config.refine_radius_sec)
            end = min(info.duration_sec, last_sec + config.refine_radius_sec)
            frames = list(iter_sampled_frames(info.path, fps=config.refine_fps,
                          max_edge=config.refine_max_edge, start_sec=start, end_sec=end))
            if not frames:
                raise InvalidModelOutputError("Cannot verify the proposed event window")
            # A different sampling grid (and its half-open end) can omit an
            # original endpoint. Reread those exact source frames at fine size.
            context_edges = [get_frame_at(info.path, timestamp, max_edge=config.refine_max_edge)
                             for timestamp in (first_sec, last_sec)]
            required_times = {frame.timestamp_sec for frame in context_edges}
            frames = sorted({frame.timestamp_sec: frame for frame in [*frames, *context_edges]}.values(),
                            key=lambda frame: frame.timestamp_sec)
            if len(frames) > config.max_refine_frames:
                required = [frame for frame in frames if frame.timestamp_sec in required_times]
                optional = [frame for frame in frames if frame.timestamp_sec not in required_times]
                slots = config.max_refine_frames - len(required)
                if slots == 1:
                    selected = [optional[len(optional) // 2]]
                elif slots > 1:
                    last = len(optional) - 1
                    selected = [optional[round(i * last / (slots - 1))] for i in range(slots)]
                else:
                    selected = []
                frames = sorted([*required, *selected], key=lambda frame: frame.timestamp_sec)
            confirmed.extend(analyze_window(frames, [self.rules[event.rule_id]], "fine"))

        violations = []
        for index, event in enumerate(merge_events(confirmed, config.merge_sec), 1):
            remaining()
            rule = self.rules[event.rule_id]
            evidence = []
            seen = set()
            anchor = get_frame_at(info.path, event.timecode_sec)
            for offset in (-0.5, 0.0, 0.5):
                frame = anchor if offset == 0 else get_frame_at(
                    info.path, min(info.duration_sec, max(0.0, anchor.timestamp_sec + offset)))
                if frame.timestamp_sec in seen:
                    continue
                seen.add(frame.timestamp_sec)
                path = output / f"event_{index:03d}_{len(evidence) + 1}.jpg"
                frame.image.save(path, format="JPEG", quality=92)
                evidence.append(EvidenceFrame(timestamp_sec=frame.timestamp_sec, image_path=path))
            violations.append(Violation(
                type=rule.title, timecode_sec=anchor.timestamp_sec,
                description=event.description, rule_id=rule.id, source=rule.source,
                evidence_frames=evidence,
            ))
        remaining()
        result = AnalysisResult(verdict="violation" if violations else "no_violation",
                                violations=violations)
        (output / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result
