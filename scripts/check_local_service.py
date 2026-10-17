"""Exercise the actual local model service with synthetic non-banking footage.

No bank videos are read. This is a runtime check, not an accuracy benchmark.
Run from the repository root after /health reports analyzer_ready=true.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import time

import av
from dotenv import dotenv_values
from PIL import Image, ImageDraw
import requests


def make_video(path: Path):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 256, 384, "yuv420p"
        for i in range(24):
            image = Image.new("RGB", (256, 384), "white")
            ImageDraw.Draw(image).ellipse((32 + i, 100, 96 + i, 164), fill="red")
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = i, Fraction(1, 4)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    # Do not accidentally send even synthetic test data or local API keys to a
    # remote destination; this helper intentionally tests loopback only.
    from urllib.parse import urlparse
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("This check is restricted to the local machine")
    config = dotenv_values(".env")
    key = config.get("CASHLENS_API_KEY", "dev-key")
    headers = {"X-API-Key": key}
    health = requests.get(args.url + "/health", timeout=5).json()
    if not health.get("analyzer_ready") or health.get("analyzer") != "model":
        raise RuntimeError("The actual local model is not ready")
    Path("local/runtime").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir="local/runtime") as tmp:
        video = Path(tmp) / "synthetic.mov"
        make_video(video)
        with video.open("rb") as source:
            response = requests.post(args.url + "/api/v1/analyze", headers=headers,
                                     files={"file": ("synthetic.mov", source, "video/quicktime")}, timeout=15)
        response.raise_for_status()
        job_id = response.json()["job_id"]
        started = time.monotonic()
        while time.monotonic() - started < args.timeout:
            response = requests.get(args.url + f"/api/v1/jobs/{job_id}", headers=headers, timeout=10)
            response.raise_for_status()
            job = response.json()
            if job["status"] == "failed":
                raise RuntimeError("Synthetic inference failed: " + job.get("error", "unknown"))
            if job["status"] == "done":
                assert job["analyzer"] == "model"
                assert job["verdict"] == "no_violation" and not job["violations"], (
                    "The model flagged synthetic geometric shapes as a banking event")
                web = requests.get(args.url + f"/web/jobs/{job_id}", timeout=5)
                web.raise_for_status()
                summary = {"synthetic_only": True, "job_id": job_id, "analyzer": job["analyzer"],
                           "verdict": job["verdict"], "event_count": len(job["violations"]),
                           "processing_sec": job["processing_time_sec"], "rules": health["rule_count"],
                           "web_result_http": web.status_code}
                Path("local/runtime/service-smoke.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary, indent=2))
                return
            time.sleep(2)
        raise TimeoutError("The local service smoke check exceeded its time budget")


if __name__ == "__main__":
    main()
