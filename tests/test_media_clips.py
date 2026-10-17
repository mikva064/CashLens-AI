"""Event MP4 bounds and failures using only generated footage."""
from pathlib import Path
import subprocess
from types import SimpleNamespace

import av
import pytest

from app import media
from cashlens_ai.video import probe_video


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    path = tmp_path_factory.mktemp("event-clips") / "synthetic.mov"
    subprocess.run([media.ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "testsrc=size=100x60:rate=10",
                    "-t", "10", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True)
    return path


@pytest.mark.parametrize("at,start,end", [(0.5, 0, 3.5), (5, 2, 8), (9.5, 6.5, 10)])
def test_event_clip_is_h264_and_clamped(source, tmp_path, at, start, end):
    target = tmp_path / "clips" / "event_001.mp4"
    assert media.make_event_clip(source, at, target, duration_sec=10) == {
        "start_sec": start, "end_sec": end}
    info = probe_video(target)
    assert info.codec == "h264"
    assert info.duration_sec == pytest.approx(end - start, abs=0.12)
    with av.open(str(target)) as container:
        frame = next(container.decode(video=0))
        assert float(frame.pts * frame.time_base) == pytest.approx(0, abs=0.01)
    assert not target.with_name("event_001.part.mp4").exists()


def test_clip_encoding_failure_cleans_partial_and_raises(source, tmp_path, monkeypatch):
    target = tmp_path / "event.mp4"
    partial = target.with_name("event.part.mp4")
    partial.write_bytes(b"partial")
    monkeypatch.setattr(media, "_run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="encode event clip"):
        media.make_event_clip(source, 5, target, duration_sec=10)
    assert not target.exists() and not partial.exists()


@pytest.mark.parametrize("at,duration", [(-1, 10), (11, 10), (0, 0), (float("nan"), 10)])
def test_invalid_clip_bounds_rejected(source, tmp_path, at, duration):
    with pytest.raises(ValueError):
        media.make_event_clip(source, at, tmp_path / "event.mp4", duration_sec=duration)
