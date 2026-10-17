"""Работа с видео через ffmpeg: длительность, mp4 для браузера, кадры-доказательства.

ffmpeg берётся из PATH, иначе из пакета imageio-ffmpeg (поставляется с бинарником).
"""

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
        ["-y", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "26", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(tmp)],
        timeout=180,
    )
    if proc.returncode != 0 or not tmp.exists():
        return False
    tmp.replace(dst)
    return True


def extract_frame(src: Path, at_sec: float, dst: Path) -> bool:
    proc = _run(
        ["-y", "-ss", f"{max(0.0, at_sec):.3f}", "-i", str(src),
         "-frames:v", "1", "-q:v", "3", str(dst)],
        timeout=30,
    )
    return proc.returncode == 0 and dst.exists()
