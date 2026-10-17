"""Send five simultaneous local uploads and verify terminal responses within 300s.

No scene descriptions or image data are printed. Without --videos, five small
synthetic clips are generated. A queue_full failure is a valid bounded response;
unexpected model errors, HTTP errors and missing terminal answers fail the test.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import urlparse

import av
from dotenv import dotenv_values
import httpx
from PIL import Image, ImageDraw


def make_clip(path: Path, index: int) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 128, 192, "yuv420p"
        for number in range(12):
            image = Image.new("RGB", (128, 192), "white")
            ImageDraw.Draw(image).rectangle((15 + number, 30, 45 + number, 60),
                                            fill=(40 * index, 60, 140))
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = number, Fraction(1, 4)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def run_one(url: str, key: str, video: Path, index: int,
            barrier: threading.Barrier, budget: float = 300) -> dict:
    barrier.wait(timeout=15)
    started = time.monotonic()
    row = {"upload": index, "job_id": None, "status": "error"}
    try:
        with httpx.Client(headers={"X-API-Key": key}, timeout=20) as client:
            with video.open("rb") as source:
                response = client.post(url + "/api/v1/analyze",
                                       files={"file": (f"load-{index}.mov", source, "video/quicktime")})
            response.raise_for_status()
            posted = response.json()
            row["job_id"] = posted["job_id"]
            # Always verify GET, including jobs rejected at admission.
            while time.monotonic() - started < budget:
                remaining = budget - (time.monotonic() - started)
                response = client.get(url + "/api/v1/jobs/" + row["job_id"],
                                      timeout=max(0.01, min(5, remaining)))
                response.raise_for_status()
                job = response.json()
                if job["status"] in {"done", "failed"}:
                    row.update(status=job["status"], failure_code=job.get("failure_code"),
                               error=job.get("error"))
                    if job["status"] == "failed" and "verdict" in job:
                        raise ValueError("A failed job must not contain a verdict")
                    break
                time.sleep(min(0.5, max(0.0, budget - (time.monotonic() - started))))
            else:
                row.update(status="timeout", error="No terminal response within the budget")
    except Exception as exc:
        # Keep source paths, HTTP bodies and any model descriptions out of logs.
        row.update(status="error", error=type(exc).__name__)
    row["elapsed_sec"] = round(time.monotonic() - started, 3)
    row["within_budget"] = row["elapsed_sec"] <= budget
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("local/load-test.json"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("This dataset test is restricted to the local computer")
    key = os.getenv("CASHLENS_API_KEY") or dotenv_values(".env").get("CASHLENS_API_KEY", "dev-key")
    health = httpx.get(args.url.rstrip("/") + "/health", timeout=5)
    health.raise_for_status()
    if not health.json().get("analyzer_ready"):
        parser.error("Wait until the local analyzer is ready")
    if args.videos and len(args.videos) not in {1, 5}:
        parser.error("Provide one video to repeat five times, or five video paths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="load-", dir=args.output.parent) as temp:
        videos = args.videos
        if not videos:
            videos = [Path(temp) / f"synthetic-{i}.mov" for i in range(5)]
            for i, path in enumerate(videos):
                make_clip(path, i)
        elif len(videos) == 1:
            videos *= 5
        if any(not path.is_file() for path in videos):
            parser.error("An input video is missing")
        barrier = threading.Barrier(5)
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(run_one, args.url.rstrip("/"), key, video, i + 1, barrier)
                       for i, video in enumerate(videos)]
            rows = [future.result() for future in futures]
    unexpected = [r for r in rows if r["status"] not in {"done", "failed"}
                  or not r["within_budget"]
                  or (r["status"] == "failed" and r.get("failure_code") != "queue_full")]
    report = {"passed": not unexpected and any(r["status"] == "done" for r in rows),
              "budget_sec": 300, "uploads": 5,
              "done": sum(r["status"] == "done" for r in rows),
              "queue_rejected": sum(r.get("failure_code") == "queue_full" for r in rows),
              "results": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
