"""Streaming local video decoding with timestamps from the source video.

Timestamps are seconds relative to the video stream start. Container rotation
metadata is not applied; coordinates and sizes describe the encoded image.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Iterator

import av
from PIL import Image

from .errors import VideoReadError


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration_sec: float
    width: int
    height: int
    codec: str
    has_audio: bool


@dataclass(frozen=True)
class SampledFrame:
    timestamp_sec: float
    image: Image.Image


def _validate_number(value: Real, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        comparison = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {comparison} number")
    return number


def _validate_max_edge(max_edge: int | None) -> None:
    if max_edge is not None and (
        isinstance(max_edge, bool) or not isinstance(max_edge, int) or max_edge < 1
    ):
        raise ValueError("max_edge must be a positive integer or None")


@contextmanager
def _open_video(path: str | Path):
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise VideoReadError(f"Video file does not exist: {source}")
    try:
        with av.open(str(source), mode="r") as container:
            if not container.streams.video:
                raise VideoReadError(f"File has no video stream: {source}")
            yield source, container, container.streams.video[0]
    except VideoReadError:
        raise
    except (av.error.FFmpegError, OSError, EOFError, ValueError) as exc:
        raise VideoReadError(f"Cannot decode video {source}: {exc}") from exc


def _raw_timestamp(frame, stream) -> float:
    time_base = frame.time_base or stream.time_base
    if frame.pts is None or time_base is None:
        raise VideoReadError("Decoded video frame has no source timestamp")
    timestamp = float(frame.pts * time_base)
    if not math.isfinite(timestamp):
        raise VideoReadError("Decoded video frame has an invalid timestamp")
    return timestamp


def _stream_origin(stream) -> float | None:
    if stream.start_time is not None and stream.time_base is not None:
        return float(stream.start_time * stream.time_base)
    return None


def _decode_frames(container, stream, origin: float | None = None):
    if origin is None:
        origin = _stream_origin(stream)
    for frame in container.decode(stream):
        raw = _raw_timestamp(frame, stream)
        if origin is None:
            origin = raw
        yield frame, max(0.0, raw - origin)


def _decode_from(path: str | Path, start: float):
    """Seek before start when stream origin is known; never renormalize a seek."""
    with _open_video(path) as (source, container, stream):
        origin = _stream_origin(stream)
        if start <= 0 or origin is None or stream.time_base is None:
            yield from _decode_frames(container, stream)
            return
        try:
            offset = int((origin + start) / float(stream.time_base))
            container.seek(offset, stream=stream, backward=True, any_frame=False)
            frames = _decode_frames(container, stream, origin)
            first = next(frames, None)
        except (av.error.FFmpegError, OSError, EOFError, ValueError, OverflowError):
            first = None
        if first is not None and first[1] <= start + 1e-9:
            yield first
            yield from frames
            return
    # Empty or forward-landing seeks cannot establish the requested prefix.
    # Reopen sequentially, as get_frame_at does for unreliable containers.
    with _open_video(path) as (source, container, stream):
        yield from _decode_frames(container, stream)


def _sample(frame, timestamp: float, max_edge: int | None) -> SampledFrame:
    image = frame.to_image().convert("RGB")
    if max_edge is not None and max(image.size) > max_edge:
        scale = max_edge / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return SampledFrame(timestamp_sec=timestamp, image=image)


def probe_video(path: str | Path) -> VideoInfo:
    """Read video metadata and verify that at least one frame can be decoded."""
    with _open_video(path) as (source, container, stream):
        frames = _decode_frames(container, stream)
        first = next(frames, None)
        if first is None:
            raise VideoReadError(f"Video contains no decodable frames: {source}")

        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        if not math.isfinite(duration) or duration <= 0:
            # Container duration can include a longer audio stream, so prefer
            # video duration whenever the demuxer supplies it.
            if container.duration is not None:
                duration = float(container.duration / av.time_base)
        if not math.isfinite(duration) or duration <= 0:
            last_frame, last_time = first
            for last_frame, last_time in frames:
                pass
            frame_duration = getattr(last_frame, "duration", 0)
            time_base = last_frame.time_base or stream.time_base
            tail = float(frame_duration * time_base) if frame_duration and time_base else 0.0
            if tail <= 0 and stream.average_rate:
                tail = 1.0 / float(stream.average_rate)
            duration = max(0.0, last_time + tail)

        return VideoInfo(
            path=source,
            duration_sec=duration,
            width=first[0].width,
            height=first[0].height,
            codec=stream.codec_context.name or "unknown",
            has_audio=bool(container.streams.audio),
        )


def iter_sampled_frames(
    path: str | Path,
    *,
    fps: float = 1.0,
    max_edge: int = 448,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> Iterator[SampledFrame]:
    """Yield RGB frames at/after a regular grid, preserving source timestamps.

    The interval is [start_sec, end_sec). A source frame is yielded at most
    once even when sampling is faster than the source. Frames stay streamed;
    the function never builds a list of the entire video. Validation happens
    when iteration starts, as is usual for Python generators.
    """
    rate = _validate_number(fps, "fps", positive=True)
    start = _validate_number(start_sec, "start_sec")
    end = None if end_sec is None else _validate_number(end_sec, "end_sec")
    _validate_max_edge(max_edge)
    if end is not None and end <= start:
        raise ValueError("end_sec must be greater than start_sec")

    decoded_any = False
    next_target = start
    for frame, timestamp in _decode_from(path, start):
        decoded_any = True
        if end is not None and timestamp + 1e-9 >= end:
            break
        if timestamp + 1e-9 < next_target:
            continue
        yield _sample(frame, timestamp, max_edge)
        # Advance past every target covered by this frame, without
        # emitting duplicate images for a low-frame-rate source.
        covered_targets = (timestamp - start) * rate
        if math.isfinite(covered_targets):
            next_index = math.floor(covered_targets + 1e-9) + 1
            next_target = start + next_index / rate
        else:
            # A finite but extremely large sampling rate may overflow
            # the multiplication; all subsequent frames still qualify.
            next_target = timestamp
    if not decoded_any:
        raise VideoReadError(f"Video contains no decodable frames: {Path(path)}")


def get_frames_at(
    path: str | Path, timestamps_sec: list[float], *, max_edge: int | None = None
) -> list[SampledFrame]:
    """Read nearest source frames in one range decode, converting only targets.

    Duplicate source frames are returned once, sorted by global video time.
    Only the previous/current decoded frames and selected RGB images are held.
    """
    targets = sorted({_validate_number(value, "timestamp_sec") for value in timestamps_sec})
    _validate_max_edge(max_edge)
    if not targets:
        return []
    selected: dict[float, SampledFrame] = {}
    target_index = 0
    previous = None
    for current in _decode_from(path, targets[0]):
        while target_index < len(targets) and current[1] >= targets[target_index]:
            target = targets[target_index]
            nearest = current
            if previous is not None and target - previous[1] <= current[1] - target + 1e-9:
                nearest = previous
            if nearest[1] not in selected:
                selected[nearest[1]] = _sample(nearest[0], nearest[1], max_edge)
            target_index += 1
        if target_index == len(targets):
            break
        previous = current
    else:
        if previous is None:
            raise VideoReadError(f"Video contains no decodable frames: {Path(path)}")
        if previous[1] not in selected:
            selected[previous[1]] = _sample(previous[0], previous[1], max_edge)
    return sorted(selected.values(), key=lambda frame: frame.timestamp_sec)


def _nearest(frames, target: float):
    previous = None
    for current in frames:
        if current[1] >= target:
            if previous is None:
                # Seeking may land after the target on some containers.
                return current, current[1] > target + 1e-9
            before = target - previous[1]
            after = current[1] - target
            return (previous if before <= after + 1e-9 else current), False
        previous = current
    return previous, previous is None


def get_frame_at(
    path: str | Path, timestamp_sec: float, *, max_edge: int | None = None
) -> SampledFrame:
    """Return the nearest source frame; ties choose the earlier frame.

    Times past the final frame return that frame. A keyframe seek limits work
    for long videos; an initial decode and sequential fallback handle files
    with missing start-time metadata or unreliable seeking.
    """
    target = _validate_number(timestamp_sec, "timestamp_sec")
    _validate_max_edge(max_edge)
    with _open_video(path) as (source, container, stream):
        first_frame = next(container.decode(stream), None)
        if first_frame is None:
            raise VideoReadError(f"Video contains no decodable frames: {source}")
        origin = _stream_origin(stream)
        if origin is None:
            origin = _raw_timestamp(first_frame, stream)
        first_time = max(0.0, _raw_timestamp(first_frame, stream) - origin)
        if target <= first_time:
            return _sample(first_frame, first_time, max_edge)

        nearest = None
        fallback = True
        if stream.time_base is not None:
            try:
                offset = int((origin + target) / float(stream.time_base))
                container.seek(offset, stream=stream, backward=True, any_frame=False)
                nearest, fallback = _nearest(_decode_frames(container, stream, origin), target)
            except (av.error.FFmpegError, OSError, EOFError, ValueError, OverflowError):
                fallback = True
        if not fallback and nearest is not None:
            return _sample(nearest[0], nearest[1], max_edge)

    # Reopen rather than relying on a second seek to reset an unreliable
    # container; the fallback also covers an empty decode following a seek.
    with _open_video(path) as (source, container, stream):
        nearest, _ = _nearest(_decode_frames(container, stream), target)
        if nearest is None:
            raise VideoReadError(f"Video contains no decodable frames: {source}")
        return _sample(nearest[0], nearest[1], max_edge)
