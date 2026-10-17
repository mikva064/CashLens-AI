"""Local developer commands. Only download-model intentionally uses the network."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from .errors import CashLensError


def _print(value):
    # ASCII output also works with the default Windows console encoding.
    print(json.dumps(value, ensure_ascii=True, indent=2, default=str))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cashlens-ai")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check the local environment, without loading weights")
    audit = commands.add_parser("audit", help="Audit labels and create a content-grouped split")
    audit.add_argument("--labels", type=Path, required=True)
    audit.add_argument("--videos", type=Path, required=True)
    audit.add_argument("--output", type=Path, default=Path("local/audit"))
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--holdout-per-class", type=int, default=5)
    review = commands.add_parser("review", help="Create a private offline annotation viewer")
    review.add_argument("--manifest", type=Path, default=Path("local/audit/manifest.json"))
    review.add_argument("--output", type=Path, default=Path("local/review.html"))
    review.add_argument("--include-holdout", action="store_true")
    inspect = commands.add_parser("inspect", help="Read local video metadata")
    inspect.add_argument("video", type=Path)
    evaluate = commands.add_parser("evaluate", help="Evaluate saved predictions by unique video")
    evaluate.add_argument("--manifest", type=Path, default=Path("local/audit/manifest.json"))
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--split", choices=["holdout", "development"], default="holdout")
    evaluate.add_argument("--output", type=Path, required=True)
    download = commands.add_parser("download-model", help="Download weights before offline inference")
    download.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
    download.add_argument("--revision", default="main")
    download.add_argument("--output", type=Path, default=Path("local/models/qwen3-vl-2b"))
    analyze = commands.add_parser("analyze", help="Analyze using local weights and verified rules")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--model-dir", type=Path, required=True)
    analyze.add_argument("--rules", type=Path, default=Path("config/rules.json"))
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--budget", type=float, default=240.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            packages = {}
            for name in ("av", "Pillow", "pydantic", "openpyxl", "torch", "torchvision",
                         "transformers", "accelerate", "bitsandbytes"):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
            gpu = {"available": False}
            if packages["torch"]:
                import torch
                gpu["available"] = torch.cuda.is_available()
                if gpu["available"]:
                    gpu.update(name=torch.cuda.get_device_name(0),
                               memory_bytes=torch.cuda.get_device_properties(0).total_memory)
            _print({"python": sys.version.split()[0], "packages": packages, "cuda": gpu})
        elif args.command == "audit":
            from .dataset import audit_dataset
            _print(audit_dataset(args.labels, args.videos, args.output, seed=args.seed,
                                 holdout_per_class=args.holdout_per_class))
        elif args.command == "review":
            from .review import build_review
            _print({"review": str(build_review(args.manifest, args.output,
                                                include_holdout=args.include_holdout))})
        elif args.command == "inspect":
            from .video import probe_video
            _print(asdict(probe_video(args.video)))
        elif args.command == "evaluate":
            from .evaluate import evaluate_predictions
            report = evaluate_predictions(args.manifest, args.predictions, split=args.split)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            _print({"report": str(args.output.resolve()), "counts": report["counts"],
                    "metrics": report["metrics"]})
        elif args.command == "download-model":
            # An explicit setup command in a separate process; inference never
            # calls this. Use a task-local cache, not the user's home directory.
            os.environ["HF_HOME"] = str(args.output.resolve() / ".hf-cache")
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            from huggingface_hub import HfApi, snapshot_download
            revision = HfApi().model_info(args.repo, revision=args.revision).sha
            location = snapshot_download(
                args.repo, revision=revision, local_dir=args.output, max_workers=2,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*"],
            )
            (args.output / "cashlens-model-revision.json").write_text(
                json.dumps({"repo": args.repo, "revision": revision}, indent=2), encoding="utf-8")
            _print({"model_dir": location, "repo": args.repo, "revision": revision})
        elif args.command == "analyze":
            from . import analyze_video
            from .pipeline import AnalyzerConfig
            result = analyze_video(args.video, model_dir=args.model_dir, rules_path=args.rules,
                                   output_dir=args.output,
                                   config=AnalyzerConfig(time_budget_sec=args.budget))
            # Full descriptions stay in the local result.json, not console logs.
            _print({"verdict": result.verdict, "event_count": len(result.violations),
                    "result_path": str(args.output.resolve() / "result.json")})
    except (CashLensError, OSError, ValueError, ImportError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
