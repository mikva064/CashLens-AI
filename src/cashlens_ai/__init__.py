"""CashLens local AI component."""

from .contracts import AnalysisResult


def analyze_video(video_path, *, model_dir, rules_path, output_dir, config=None,
                  observations_path=None):
    """Analyze one video; raises CashLensError on unavailable/incomplete analysis.

    For a server worker, reuse VideoAnalyzer + LocalQwenBackend rather than loading
    model weights for each job. Serialize GPU jobs in that worker.

    The observation catalog defaults to observations.json beside rules_path. A
    missing catalog is an error, so a partial configuration cannot silently omit
    review findings. Supply observations_path for a different local catalog.
    """
    from pathlib import Path

    from .backend import LocalQwenBackend
    from .observations import load_observations
    from .pipeline import VideoAnalyzer
    from .rules import load_rules

    rules = load_rules(rules_path)
    observations = load_observations(observations_path if observations_path is not None else
                                     Path(rules_path).with_name("observations.json"))
    backend = LocalQwenBackend(model_dir)
    try:
        return VideoAnalyzer(backend, rules, observations=observations,
                             config=config).analyze(video_path, output_dir)
    finally:
        backend.close()


__all__ = ["AnalysisResult", "analyze_video"]
