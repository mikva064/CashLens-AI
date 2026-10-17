"""Public local entry points retain review findings without loading a real model."""
import json
from pathlib import Path

import pytest

import cashlens_ai
from cashlens_ai import backend, cli, pipeline
from cashlens_ai.contracts import AnalysisResult, EvidenceFrame, Observation
from cashlens_ai.errors import InvalidModelOutputError, RuleValidationError


def write_catalogs(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps([{
        "id": "test_rule", "title": "Synthetic rule", "verified": True,
        "observable_condition": "A synthetic visible action with sufficient context.",
        "source": {"document": "Synthetic source", "section": "1",
                   "url": "https://example.invalid/rule"},
    }]), encoding="utf-8")
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(json.dumps([{
        "id": "test_observation", "title": "Synthetic observation",
        "observable_condition": "A synthetic action requiring a context review.",
        "description_template": "A visible action needs review.",
        "review_reason": "Additional transaction context must be reviewed.",
    }]), encoding="utf-8")
    return rules_path, observations_path


def review_result():
    return AnalysisResult(verdict="requires_review", violations=[], observations=[Observation(
        observation_id="test_observation", type="Synthetic observation", timecode_sec=1,
        description="Private synthetic description must stay out of the console.",
        evidence_frames=[EvidenceFrame(timestamp_sec=1, image_path=Path("synthetic.jpg"))],
        review_reason="Additional transaction context must be reviewed.",
    )])


@pytest.mark.parametrize("explicit_catalog", [False, True])
def test_helper_loads_observations_and_closes_backend(tmp_path, monkeypatch, explicit_catalog):
    rules_path, observations_path = write_catalogs(tmp_path)
    if explicit_catalog:
        renamed = tmp_path / "chosen_catalog.json"
        observations_path.rename(renamed)
        observations_path = renamed
    calls = {}

    class Backend:
        def __init__(self, path):
            calls["model"] = path

        def close(self):
            calls["closed"] = True

    class Analyzer:
        def __init__(self, instance, rules, *, observations, config):
            calls.update(rules=rules, observations=observations, config=config)

        def analyze(self, video, output):
            calls.update(video=video, output=output)
            return review_result()

    monkeypatch.setattr(backend, "LocalQwenBackend", Backend)
    monkeypatch.setattr(pipeline, "VideoAnalyzer", Analyzer)
    kwargs = {"observations_path": observations_path} if explicit_catalog else {}
    result = cashlens_ai.analyze_video("synthetic.mov", model_dir="local-weights", rules_path=rules_path,
                                      output_dir=tmp_path / "output", **kwargs)
    assert result.verdict == "requires_review"
    assert [item.id for item in calls["observations"]] == ["test_observation"]
    assert [item.id for item in calls["rules"]] == ["test_rule"]
    assert calls["closed"] and calls["video"] == "synthetic.mov"


def test_missing_default_catalog_fails_before_model_construction(tmp_path, monkeypatch):
    rules_path, observations_path = write_catalogs(tmp_path)
    observations_path.unlink()

    def unexpected_model(*args, **kwargs):
        raise AssertionError("No model should be constructed with an incomplete catalog")

    monkeypatch.setattr(backend, "LocalQwenBackend", unexpected_model)
    with pytest.raises(RuleValidationError):
        cashlens_ai.analyze_video("synthetic.mov", model_dir="local-weights", rules_path=rules_path,
                                  output_dir=tmp_path / "output")


def test_helper_closes_backend_when_analysis_fails(tmp_path, monkeypatch):
    rules_path, _ = write_catalogs(tmp_path)
    closed = []

    class Backend:
        def __init__(self, path):
            pass

        def close(self):
            closed.append(True)

    class Analyzer:
        def __init__(self, *args, **kwargs):
            pass

        def analyze(self, *args):
            raise InvalidModelOutputError("Synthetic failure")

    monkeypatch.setattr(backend, "LocalQwenBackend", Backend)
    monkeypatch.setattr(pipeline, "VideoAnalyzer", Analyzer)
    with pytest.raises(InvalidModelOutputError):
        cashlens_ai.analyze_video("synthetic.mov", model_dir="local-weights", rules_path=rules_path,
                                  output_dir=tmp_path / "output")
    assert closed == [True]


@pytest.mark.parametrize("explicit_catalog", [False, True])
def test_cli_preserves_review_verdict_and_prints_only_counts(tmp_path, monkeypatch, capsys,
                                                            explicit_catalog):
    calls = {}

    def analyze(video, **kwargs):
        calls.update(video=video, **kwargs)
        return review_result()

    monkeypatch.setattr(cashlens_ai, "analyze_video", analyze)
    arguments = ["analyze", "synthetic.mov", "--model-dir", "local-weights",
                 "--rules", "custom/rules.json", "--output", str(tmp_path / "output")]
    if explicit_catalog:
        arguments += ["--observations", "custom/chosen_catalog.json"]
    assert cli.main(arguments) == 0
    console = capsys.readouterr()
    payload = json.loads(console.out)
    assert payload["verdict"] == "requires_review"
    assert payload["event_count"] == 0 and payload["observation_count"] == 1
    assert payload["review_required"] is True
    assert "Private synthetic description" not in console.out
    assert calls["observations_path"] == (Path("custom/chosen_catalog.json") if explicit_catalog else None)


def test_cli_missing_catalog_is_a_clear_failure(tmp_path, capsys):
    rules_path, observations_path = write_catalogs(tmp_path)
    observations_path.unlink()
    assert cli.main(["analyze", "synthetic.mov", "--model-dir", "local-weights",
                     "--rules", str(rules_path), "--output", str(tmp_path / "output")]) == 1
    console = capsys.readouterr()
    assert console.out == ""
    assert "RuleValidationError" in console.err and "observation" in console.err.lower()
