"""Small generated MOV fixtures exercise decoding without using private data."""

from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
from PIL import Image

from cashlens_ai.errors import VideoReadError
from cashlens_ai import video
from cashlens_ai.video import get_frame_at, get_frames_at, iter_sampled_frames, probe_video


def make_mov(path: Path, timestamps=(20, 22, 25, 29)) -> None:
    """Encode nonuniform presentation times and a nonzero stream start."""
    with av.open(str(path), mode="w", format="mov") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width = 80
        stream.height = 40
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 10)
        stream.codec_context.gop_size = 2
        for index, pts in enumerate(timestamps):
            image = Image.new("RGB", (80, 40), (index * 60, 20, 20))
            frame = av.VideoFrame.from_image(image)
            frame.pts = pts
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


class VideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "variable-start.mov"
        make_mov(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_probe_uses_video_metadata(self):
        info = probe_video(self.path)
        self.assertEqual(info.path, self.path.resolve())
        self.assertEqual((info.width, info.height), (80, 40))
        self.assertEqual(info.codec, "mpeg4")
        self.assertFalse(info.has_audio)
        self.assertAlmostEqual(info.duration_sec, 1.0, places=3)

    def test_preserves_normalized_source_timestamps(self):
        frames = list(iter_sampled_frames(self.path, fps=100))
        self.assertEqual(len(frames), 4)
        for frame, expected in zip(frames, (0.0, 0.2, 0.5, 0.9)):
            self.assertAlmostEqual(frame.timestamp_sec, expected, places=6)
        self.assertTrue(all(frame.image.mode == "RGB" for frame in frames))

    def test_grid_sampling_does_not_fabricate_or_duplicate_frames(self):
        frames = list(iter_sampled_frames(self.path, fps=4, max_edge=30))
        self.assertEqual([round(frame.timestamp_sec, 4) for frame in frames], [0.0, 0.5, 0.9])
        self.assertTrue(all(frame.image.size == (30, 15) for frame in frames))
        self.assertEqual(len(list(iter_sampled_frames(self.path, fps=1000))), 4)

    def test_half_open_interval_and_start_past_end(self):
        frames = list(iter_sampled_frames(self.path, fps=10, start_sec=0.1, end_sec=0.9))
        self.assertEqual([round(frame.timestamp_sec, 4) for frame in frames], [0.2, 0.5])
        self.assertEqual(list(iter_sampled_frames(self.path, start_sec=10)), [])

    def test_range_seek_keeps_original_stream_origin(self):
        with patch("cashlens_ai.video._decode_frames", wraps=video._decode_frames) as decode:
            frames = list(iter_sampled_frames(self.path, fps=10, start_sec=0.5, end_sec=1.0))
        self.assertEqual([round(frame.timestamp_sec, 6) for frame in frames], [0.5, 0.9])
        self.assertEqual(decode.call_args.args[2], 2.0)

    def test_range_seek_falls_back_when_landing_after_requested_start(self):
        real_decode = video._decode_frames
        attempts = 0

        def forward_once(container, stream, origin=None):
            nonlocal attempts
            attempts += 1
            frames = real_decode(container, stream, origin)
            return (item for item in frames if item[1] >= 0.5) if attempts == 1 else frames

        with patch("cashlens_ai.video._decode_frames", side_effect=forward_once):
            frames = list(iter_sampled_frames(self.path, fps=10, start_sec=0.1, end_sec=0.9))
        self.assertEqual(attempts, 2)
        self.assertEqual([round(frame.timestamp_sec, 6) for frame in frames], [0.2, 0.5])

    def test_range_without_origin_metadata_uses_sequential_timestamps(self):
        with patch("cashlens_ai.video._stream_origin", return_value=None):
            frames = list(iter_sampled_frames(self.path, fps=10, start_sec=0.1, end_sec=0.9))
        self.assertEqual([round(frame.timestamp_sec, 6) for frame in frames], [0.2, 0.5])

    def test_batch_nearest_materializes_only_unique_selected_source_frames(self):
        with patch("cashlens_ai.video._sample", wraps=video._sample) as convert:
            frames = get_frames_at(self.path, [0.9, 0.18, 0.35, 0.5, 0.5], max_edge=20)
        self.assertEqual([round(frame.timestamp_sec, 6) for frame in frames], [0.2, 0.5, 0.9])
        self.assertEqual(convert.call_count, 3)
        self.assertTrue(all(frame.image.size == (20, 10) for frame in frames))
        self.assertAlmostEqual(get_frames_at(self.path, [100])[0].timestamp_sec, 0.9)

    def test_nearest_frame_and_video_bounds(self):
        for target, expected in ((0.0, 0.0), (0.18, 0.2), (0.35, 0.2), (0.37, 0.5), (100.0, 0.9), (1e308, 0.9)):
            with self.subTest(target=target):
                frame = get_frame_at(self.path, target)
                self.assertAlmostEqual(frame.timestamp_sec, expected, places=6)
                self.assertEqual(frame.image.size, (80, 40))
        self.assertEqual(get_frame_at(self.path, 0.5, max_edge=20).image.size, (20, 10))

    def test_nearest_frame_falls_back_when_seek_decode_is_empty(self):
        # An empty post-seek decode should cause a clean sequential reopen.
        real_nearest = video._nearest
        attempts = 0

        def empty_then_real(frames, target):
            nonlocal attempts
            attempts += 1
            return (None, True) if attempts == 1 else real_nearest(frames, target)

        with patch("cashlens_ai.video._nearest", side_effect=empty_then_real) as nearest:
            result = get_frame_at(self.path, 0.5)
        self.assertAlmostEqual(result.timestamp_sec, 0.5, places=6)
        self.assertEqual(result.image.size, (80, 40))
        self.assertEqual(nearest.call_count, 2)

    def test_invalid_sampling_parameters(self):
        for params in (
            {"fps": 0}, {"fps": -1}, {"fps": float("nan")}, {"fps": float("inf")},
            {"fps": True}, {"max_edge": 0}, {"max_edge": 3.5},
            {"start_sec": -0.1}, {"end_sec": 0}, {"start_sec": 1, "end_sec": 0.5},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                list(iter_sampled_frames(self.path, **params))
        for target in (-1, float("nan"), float("inf"), True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                get_frame_at(self.path, target)

    def test_missing_and_corrupt_files(self):
        corrupt = Path(self.temp.name) / "corrupt.mov"
        corrupt.write_bytes(b"This is not a MOV video")
        empty = Path(self.temp.name) / "empty.mov"
        empty.write_bytes(b"")
        for path in (Path(self.temp.name) / "absent.mov", corrupt, empty):
            with self.subTest(path=path):
                with self.assertRaises(VideoReadError):
                    probe_video(path)
                with self.assertRaises(VideoReadError):
                    list(iter_sampled_frames(path))
                with self.assertRaises(VideoReadError):
                    get_frame_at(path, 0)


if __name__ == "__main__":
    unittest.main()
