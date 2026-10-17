"""Real container rotation, synthetic asymmetric pixels, no customer footage."""
from fractions import Fraction
import subprocess

import av
from PIL import Image
import pytest

from app import media
from cashlens_ai.video import get_frame_at, get_frames_at, iter_sampled_frames, probe_video


@pytest.fixture
def rotated_video(tmp_path):
    raw = tmp_path / "encoded.mov"
    image = Image.new("RGB", (80, 40), "red")
    image.paste((0, 0, 255), (40, 0, 80, 40))
    with av.open(str(raw), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height = image.size
        stream.pix_fmt = "yuv420p"
        for index in range(20):
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    rotated = tmp_path / "rotate90.mov"
    # FFmpeg >= 7 no longer writes a display matrix via the legacy rotate tag.
    command = [media.ffmpeg_exe(), "-y", "-display_rotation:v:0", "90", "-i", str(raw),
               "-c", "copy", str(rotated)]
    remux = subprocess.run(command, capture_output=True)
    if remux.returncode:
        # Older FFmpeg uses the legacy tag to write the same MOV display matrix.
        subprocess.run([media.ffmpeg_exe(), "-y", "-i", str(raw), "-c", "copy",
                        "-metadata:s:v:0", "rotate=90", str(rotated)],
                       check=True, capture_output=True)
    with av.open(str(rotated)) as source:
        frame = next(source.decode(video=0))
        assert (frame.width, frame.height, frame.rotation) == (80, 40, 90)
    return rotated


def assert_upright(image):
    assert image.size == (40, 80)
    top = image.getpixel((20, 10))
    bottom = image.getpixel((20, 70))
    assert top[2] > top[0] + 100  # encoded right/blue becomes displayed top
    assert bottom[0] > bottom[2] + 100


def test_rotation_applies_to_probe_sampling_seeks_and_batch(rotated_video):
    info = probe_video(rotated_video)
    assert (info.width, info.height) == (40, 80)
    samples = list(iter_sampled_frames(rotated_video, fps=2))
    assert [round(frame.timestamp_sec, 2) for frame in samples] == [0, 0.5, 1, 1.5]
    for sample in samples + get_frames_at(rotated_video, [0, 0.5, 1.5]):
        assert_upright(sample.image)
    assert_upright(get_frame_at(rotated_video, 1.5).image)
    assert get_frame_at(rotated_video, 1, max_edge=20).image.size == (10, 20)


def test_ffmpeg_frame_and_preview_match_pyav_without_double_rotation(rotated_video, tmp_path):
    still = tmp_path / "frame.jpg"
    preview = tmp_path / "preview.mp4"
    assert media.extract_frame(rotated_video, 0.5, still)
    with Image.open(still) as image:
        assert_upright(image)
    assert media.to_browser_mp4(rotated_video, preview)
    with av.open(str(preview)) as source:
        frame = next(source.decode(video=0))
        assert frame.rotation == 0
        assert_upright(frame.to_image())
    assert_upright(get_frame_at(preview, 0.5).image)


def test_event_clip_applies_rotation_once(rotated_video, tmp_path):
    target = tmp_path / "clip.mp4"
    assert media.make_event_clip(rotated_video, 1, target, duration_sec=2) == {
        "start_sec": 0, "end_sec": 2}
    assert_upright(get_frame_at(target, 0.5).image)
