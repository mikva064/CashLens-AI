"""Coarse-to-fine local analysis with validated rule IDs and source evidence."""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from pydantic import Field, StrictFloat, ValidationError

from .contracts import (AnalysisResult, EvidenceFrame, Observation, ObservationDefinition,
                        Rule, StrictModel, Violation)
from .observations import category_description
from .errors import AnalysisTimeoutError, InvalidModelOutputError, RuleValidationError, VideoReadError
from .video import SampledFrame, get_frames_at, iter_sampled_frames, probe_video


class ModelEvent(StrictModel):
    rule_id: str
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1, max_length=1600)
    evidence_timestamps_sec: list[StrictFloat] | None = Field(default=None, min_length=3, max_length=3)
    cash_context_timestamp_sec: StrictFloat | None = Field(default=None, ge=0)


class ModelResponse(StrictModel):
    events: list[ModelEvent] = Field(max_length=32)


class WindowBackend(Protocol):
    def describe(self, frames: Sequence[SampledFrame], rules: list[dict], *,
                 phase: str = "coarse", timeout_sec: float | None = None,
                 candidate_timecode_sec: float | None = None,
                 candidate_timecodes_sec: list[float] | None = None,
                 candidate_radius_sec: float | None = None) -> str: ...


@dataclass(frozen=True)
class AnalyzerConfig:
    sample_fps: float = 1.0
    window_sec: float = 24.0
    overlap_sec: float = 4.0
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
        if self.max_edge < 32 or self.refine_max_edge < 32 or self.max_refine_frames < 3:
            raise ValueError("Both max_edge values >= 32 and max_refine_frames >= 3 are required")


def parse_model_response(text: str, *, allowed_ids: set[str],
                         first_sec: float, last_sec: float,
                         require_proof: bool = False,
                         frame_timestamps: Sequence[float] | None = None,
                         candidate_timecode_sec: float | None = None,
                         candidate_timecodes_sec: list[float] | None = None,
                         candidate_radius_sec: float | None = None) -> list[ModelEvent]:
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
        raise InvalidModelOutputError("Model response is not valid event JSON",
                                      reason_code="invalid_event_json") from exc
    if require_proof and len(result.events) > 3:
        raise InvalidModelOutputError("Fine verification permits at most three grounded events",
                                      reason_code="too_many_events")
    for event in result.events:
        if event.rule_id not in allowed_ids:
            raise InvalidModelOutputError("Model returned an unknown rule ID", reason_code="unknown_category")
        if not first_sec - 0.001 <= event.timecode_sec <= last_sec + 0.001:
            raise InvalidModelOutputError("Model timestamp is outside the supplied frames",
                                          reason_code="event_outside_window")
        if require_proof:
            proof = event.evidence_timestamps_sec
            if (proof is None or frame_timestamps is None
                    or any(timestamp not in frame_timestamps for timestamp in proof)
                    or not proof[0] < proof[1] < proof[2]
                    or event.timecode_sec != proof[1]
                    or event.cash_context_timestamp_sec not in proof):
                raise InvalidModelOutputError("Fine verification requires ordered source-frame proof and cash context",
                                              reason_code="invalid_source_proof")
            points = candidate_timecodes_sec if candidate_timecodes_sec is not None else [candidate_timecode_sec]
            if (not points or any(point is None for point in points) or candidate_radius_sec is None
                    or not any(abs(event.timecode_sec - point) <= candidate_radius_sec + 1e-9 for point in points)):
                raise InvalidModelOutputError("Fine action is outside the original candidate focus",
                                              reason_code="candidate_focus")
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
                 observations: list[ObservationDefinition] | None = None,
                 config: AnalyzerConfig | None = None,
                 clock: Callable[[], float] = time.monotonic):
        if not rules or any(not r.verified for r in rules):
            raise RuleValidationError("A nonempty, verified rule registry is required")
        if len({r.id for r in rules}) != len(rules):
            raise RuleValidationError("Duplicate rule IDs")
        self.backend = backend
        self.rules = {r.id: r for r in rules}
        self.observations = {item.id: item for item in (observations or [])}
        if len(self.observations) != len(observations or []) or set(self.rules) & set(self.observations):
            raise RuleValidationError("Visual category IDs must be unique across both catalogs")
        self.categories = {**self.rules, **self.observations}
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

        def analyze_window(frames, definitions, phase, *, candidates=None):
            focus = ({"candidate_timecodes_sec": [item.timecode_sec for item in candidates],
                      "candidate_radius_sec": config.refine_radius_sec}
                     if candidates is not None else {})
            raw = self.backend.describe(
                frames, [{**item.model_dump(mode="json"), "description_template": category_description(item)}
                         for item in definitions],
                phase=phase, timeout_sec=remaining(), **focus,
            )
            remaining()
            return parse_model_response(
                raw, allowed_ids={item.id for item in definitions},
                first_sec=frames[0].timestamp_sec, last_sec=frames[-1].timestamp_sec,
                require_proof=phase == "fine",
                frame_timestamps=[frame.timestamp_sec for frame in frames], **focus,
            )

        frames_per_window = max(2, round(config.window_sec * config.sample_fps))
        overlap = min(frames_per_window - 1, round(config.overlap_sec * config.sample_fps))
        buffer: list[SampledFrame] = []
        fresh = 0
        windows_analyzed = 0
        candidates: list[ModelEvent] = []
        candidate_context: dict[tuple[str, float], tuple[float, float]] = {}

        def collect_candidates(frames):
            for event in analyze_window(frames, list(self.categories.values()), "coarse"):
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
        grouped: dict[tuple[float, float], list[ModelEvent]] = {}
        for event in candidates:
            context = candidate_context[(event.rule_id, event.timecode_sec)]
            grouped.setdefault(context, []).append(event)
        for (first_sec, last_sec), context_candidates in grouped.items():
            remaining()
            # Verification must retain the before/after context that produced
            # the candidate, including service/transaction boundaries.
            start = max(0.0, first_sec - config.refine_radius_sec)
            end = min(info.duration_sec, last_sec + config.refine_radius_sec)
            # Choose target times before RGB conversion: do not materialize an
            # entire high-resolution 3-fps window just to discard most images.
            anchors = sorted({event.timecode_sec for event in context_candidates})
            required = sorted({first_sec, last_sec, *anchors})
            if len(required) > config.max_refine_frames:
                # Never silently discard a candidate or proof context to fit a cap.
                raise InvalidModelOutputError("Refinement cap cannot retain all candidate anchors",
                                              reason_code="refinement_frame_cap")
            grid_count = max(1, math.ceil((end - start) * config.refine_fps - 1e-9))
            optional = [start + index / config.refine_fps for index in range(grid_count)
                        if all(abs(start + index / config.refine_fps - anchor) > 1e-9
                               for anchor in required)]
            slots = config.max_refine_frames - len(required)
            if len(optional) > slots:
                if slots == 1:
                    optional = [optional[len(optional) // 2]]
                elif slots > 1:
                    last_index = len(optional) - 1
                    optional = [optional[round(index * last_index / (slots - 1))]
                                for index in range(slots)]
                else:
                    optional = []
            frames = get_frames_at(info.path, [*required, *optional], max_edge=config.refine_max_edge)
            if not frames:
                raise InvalidModelOutputError("Cannot verify the proposed event window",
                                              reason_code="missing_proof_context")
            confirmed.extend(analyze_window(frames, list(self.categories.values()), "fine",
                                            candidates=context_candidates))

        violations, observations = [], []
        for index, event in enumerate(merge_events(confirmed, config.merge_sec), 1):
            remaining()
            definition = self.categories[event.rule_id]
            evidence = []
            proof_frames = get_frames_at(info.path, event.evidence_timestamps_sec)
            if [frame.timestamp_sec for frame in proof_frames] != event.evidence_timestamps_sec:
                raise InvalidModelOutputError("Cannot recover the exact verified proof frames",
                                              reason_code="proof_frame_recovery")
            for frame in proof_frames:
                path = output / f"event_{index:03d}_{len(evidence) + 1}.jpg"
                frame.image.save(path, format="JPEG", quality=92)
                evidence.append(EvidenceFrame(timestamp_sec=frame.timestamp_sec, image_path=path))
            common = {"type": definition.title, "timecode_sec": event.timecode_sec,
                      "description": category_description(definition), "evidence_frames": evidence}
            if event.rule_id in self.rules:
                violations.append(Violation(**common, rule_id=definition.id, source=definition.source))
            else:
                observations.append(Observation(**common, observation_id=definition.id,
                                                review_reason=definition.review_reason))
        remaining()
        result = AnalysisResult(verdict=("violation" if violations else
                                        "requires_review" if observations else "no_violation"),
                                violations=violations, observations=observations)
        (output / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result
