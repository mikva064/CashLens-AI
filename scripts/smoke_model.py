"""Measure local generation on synthetic images, without opening bank footage.

Run from an installed checkout:
  python scripts/smoke_model.py --model-dir local/models/qwen3-vl-2b
This checks runtime compatibility only, not detection quality or the five-minute SLA.
"""

import argparse
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw

from cashlens_ai.backend import LocalQwenBackend
from cashlens_ai.pipeline import parse_model_response
from cashlens_ai.video import SampledFrame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    frames = []
    for i in range(args.frames):
        image = Image.new("RGB", (256, 256), "white")
        ImageDraw.Draw(image).rectangle((32 + i * 2, 64, 128 + i * 2, 160), fill="red")
        frames.append(SampledFrame(timestamp_sec=float(i), image=image))
    # Synthetic visual probe only. This is deliberately not a legal rule and is
    # never added to the production rules registry or reported as bank analysis.
    rules = [{"id": "synthetic_red_square", "title": "Synthetic visual probe",
              "observable_condition": "A clearly visible red square is present in the image."}]
    backend = LocalQwenBackend(args.model_dir, quantize_4bit=not args.no_quantization,
                               max_new_tokens=256, max_pixels=65536)
    try:
        started = time.monotonic()
        backend.load()
        loaded = time.monotonic()
        raw = backend.describe(frames, rules, timeout_sec=120)
        finished = time.monotonic()
        events = parse_model_response(raw, allowed_ids={rules[0]["id"]},
                                      first_sec=0, last_sec=float(args.frames - 1))
        import torch
        print(json.dumps({"synthetic_only": True, "model_load_sec": round(loaded - started, 3),
                          "generation_sec": round(finished - loaded, 3), "json_valid": True,
                          "event_count": len(events), "frames": args.frames,
                          "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
