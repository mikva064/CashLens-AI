"""CashLens local AI component."""

from .contracts import AnalysisResult


def analyze_video(video_path, *, model_dir, rules_path, output_dir, config=None):
    """Analyze one video; raises CashLensError on unavailable/incomplete analysis.

    For a server worker, reuse VideoAnalyzer + LocalQwenBackend rather than loading
    model weights for each job. Serialize GPU jobs in that worker.
    """
    from .backend import LocalQwenBackend
    from .pipeline import VideoAnalyzer
    from .rules import load_rules

    rules = load_rules(rules_path)
    backend = LocalQwenBackend(model_dir)
    try:
        return VideoAnalyzer(backend, rules, config=config).analyze(video_path, output_dir)
    finally:
        backend.close()


__all__ = ["AnalysisResult", "analyze_video"]
