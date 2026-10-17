"""Работа с видео через ffmpeg: длительность, mp4 для браузера, кадры-доказательства.

ffmpeg берётся из PATH, иначе из пакета imageio-ffmpeg (поставляется с бинарником).
"""

import math
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [ffmpeg_exe(), "-hide_banner", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def probe_duration(path: str | Path) -> Optional[float]:
    """Длительность в секундах или None, если файл не читается как видео."""
    proc = _run(["-i", str(path)], timeout=30)  # без выхода ffmpeg печатает метаданные в stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not m or "Video:" not in proc.stderr:
        return None
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def to_browser_mp4(src: Path, dst: Path) -> bool:
    """H.264 mp4 для встроенного плеера (.mov играет не во всех браузерах).

    Пишем во временный файл и переименовываем в конце, чтобы /media не отдал
    недописанное видео.
    """
    tmp = dst.with_name(dst.stem + ".part.mp4")
    proc = _run(
        ["-y", "-autorotate", "-i", str(src), "-map", "0:v:0",
         "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "26", "-pix_fmt", "yuv420p", "-an",
         "-metadata:s:v:0", "rotate=0", "-movflags", "+faststart", str(tmp)],
        timeout=180,
    )
    if proc.returncode != 0 or not tmp.exists():
        return False
    tmp.replace(dst)
    return True


def extract_frame(src: Path, at_sec: float, dst: Path) -> bool:
    proc = _run(
        ["-y", "-ss", f"{max(0.0, at_sec):.3f}", "-autorotate", "-i", str(src),
         "-map", "0:v:0",
         "-frames:v", "1", "-q:v", "3", str(dst)],
        timeout=30,
    )
    return proc.returncode == 0 and dst.exists()


def make_event_clip(src: Path, at_sec: float, dst: Path, *, duration_sec: float,
                    radius_sec: float = 3.0) -> dict[str, float]:
    """Encode an upright H.264 event clip, clamped to the source video bounds.

    Called inside the model process so the parent's job watchdog also owns
    this FFmpeg process. A missing clip is an error, never a clean verdict.
    """
    if (not all(math.isfinite(value) for value in (at_sec, duration_sec, radius_sec))
            or duration_sec <= 0 or radius_sec <= 0 or not 0 <= at_sec <= duration_sec):
        raise ValueError("Invalid event clip time bounds")
    start, end = max(0.0, at_sec - radius_sec), min(duration_sec, at_sec + radius_sec)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    try:
        proc = _run(
            ["-y", "-ss", f"{start:.6f}", "-autorotate", "-i", str(src),
             "-t", f"{end - start:.6f}", "-map", "0:v:0", "-an",
             "-vf", "scale=w='min(iw,960)':h='min(ih,960)':force_original_aspect_ratio=decrease:force_divisible_by=2",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
             "-threads", "2", "-pix_fmt", "yuv420p", "-metadata:s:v:0", "rotate=0",
             "-movflags", "+faststart", str(tmp)], timeout=60,
        )
        if proc.returncode or not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("Could not encode event clip")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return {"start_sec": start, "end_sec": end}
