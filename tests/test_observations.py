"""Separate visible observations from qualified rules, using synthetic video."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cashlens_ai.contracts import AnalysisResult, EvidenceFrame, Observation, ObservationDefinition, Rule, RuleSource, Violation
from cashlens_ai.errors import CashLensError, InvalidModelOutputError, RuleValidationError
from cashlens_ai.evaluate import evaluate_predictions
from cashlens_ai.integration import to_api_payload
from cashlens_ai.observations import catalog_hash, load_observations
from cashlens_ai.pipeline import AnalyzerConfig, VideoAnalyzer
from test_pipeline import RecordingBackend, encode_fixture, event, proof_event, response


@pytest.fixture
def definition():
    return ObservationDefinition(id="synthetic_observation", title="Synthetic observable action",
        observable_condition="The complete synthetic transition is visible before, during and after.",
        description_template="Наблюдается тестовое действие; правовая квалификация не установлена.",
        review_reason="Synthetic review requires facts that are outside the visible sequence.")


@pytest.fixture
def rule():
    return Rule(id="synthetic_event", title="Synthetic qualified event",
                observable_condition="The synthetic qualified transition is visible in full.", verified=True,
                source=RuleSource(document="Synthetic test rule", section="1", url="https://example.invalid/rule"))


def observation(definition):
    return Observation(observation_id=definition.id, type=definition.title, timecode_sec=1.,
        description=definition.description_template, review_reason=definition.review_reason,
        evidence_frames=[EvidenceFrame(timestamp_sec=1., image_path=Path("synthetic.jpg"))])


def test_observation_verdict_invariants_and_no_invented_source(definition, rule):
    item = observation(definition)
    assert AnalysisResult(verdict="requires_review", violations=[], observations=[item]).verdict == "requires_review"
    qualified = Violation(rule_id=rule.id, type=rule.title, timecode_sec=1., description="Synthetic qualified event",
                          source=rule.source, evidence_frames=item.evidence_frames)
    mixed = AnalysisResult(verdict="violation", violations=[qualified], observations=[item])
    assert mixed.violations[0].source == rule.source
    assert "source" not in item.model_dump()
    with pytest.raises(ValidationError):
        Observation.model_validate({**item.model_dump(), "source": rule.source.model_dump()})
    for verdict, violations, observations in (("no_violation", [], [item]), ("requires_review", [], []),
                                               ("requires_review", [qualified], [item]), ("violation", [], [item])):
        with pytest.raises(ValidationError):
            AnalysisResult(verdict=verdict, violations=violations, observations=observations)
    assert AnalysisResult(verdict="no_violation", violations=[]).observations == []


def test_review_result_cannot_be_exported_as_binary_api_verdict(definition, tmp_path):
    result = AnalysisResult(verdict="requires_review", violations=[], observations=[observation(definition)])
    with pytest.raises(CashLensError, match="binary verdict"):
        to_api_payload(result, job_id="synthetic", evidence_root=tmp_path / "not-created")


def test_catalog_load_is_separate_from_legal_sources_and_hash_is_stable(rule):
    catalog_path = Path(__file__).resolve().parent.parent / "config" / "observations.json"
    definitions = load_observations(catalog_path)
    assert len(definitions) == 3
    assert {item.id for item in definitions} == {"cash_split_remainder_returned", "cash_receipt_id_not_visible",
                                               "received_cash_handed_other_person"}
    assert all("source" not in item.model_dump() for item in definitions)
    assert catalog_hash([rule], definitions) == catalog_hash([rule], list(reversed(definitions)))
    changed = definitions[0].model_copy(update={"observable_condition": "Different synthetic visible condition"})
    assert catalog_hash([rule], [changed, *definitions[1:]]) != catalog_hash([rule], definitions)


def test_catalog_duplicate_ids_or_legal_source_injection_are_rejected(definition, tmp_path):
    path = tmp_path / "catalog.json"
    for data in ([definition.model_dump()] * 2,
                 [{**definition.model_dump(), "source": {"url": "https://example.invalid"}}]):
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(RuleValidationError):
            load_observations(path)


@pytest.mark.parametrize("mode", ["observation", "mixed", "corrected"])
def test_pipeline_batches_candidates_and_separates_review_from_qualified_events(tmp_path, rule, definition, mode):
    video = tmp_path / "synthetic.mov"
    encode_fixture(video)
    first = True

    def detect(phase, frames, categories):
        nonlocal first
        if phase == "coarse":
            if not first:
                return response()
            first = False
            if mode == "corrected":
                return response(event(1.0, rule_id=rule.id))
            return response(event(1.0, rule_id=definition.id), event(1.6, rule_id=rule.id))
        assert {item["id"] for item in categories} == {rule.id, definition.id}
        item = proof_event(frames, action_sec=1.0)
        item["rule_id"] = definition.id
        outputs = [item]
        if mode == "mixed":
            outputs.append(proof_event(frames, action_sec=1.6))
        return response(*outputs)

    backend = RecordingBackend(detect)
    result = VideoAnalyzer(backend, [rule], observations=[definition],
        config=AnalyzerConfig(sample_fps=2, window_sec=3, overlap_sec=0.5, max_refine_frames=8,
                              max_edge=64, refine_max_edge=80)).analyze(video, tmp_path / "result")
    assert result.verdict == ("violation" if mode == "mixed" else "requires_review")
    assert len(result.observations) == 1
    assert result.observations[0].observation_id == definition.id
    assert result.observations[0].description == definition.description_template
    assert "source" not in result.observations[0].model_dump()
    assert all(frame.image_path.is_file() for frame in result.observations[0].evidence_frames)
    assert len([call for call in backend.calls if call["phase"] == "fine"]) == 1
    if mode == "mixed":
        assert result.violations[0].source == rule.source
    else:
        assert result.violations == []


def test_unknown_observation_or_invalid_proof_fails_closed(tmp_path, rule, definition):
    video = tmp_path / "synthetic.mov"
    encode_fixture(video)
    for index, invalid in enumerate(("unknown", "proof")):
        first = True

        def detect(phase, frames, categories):
            nonlocal first
            if phase == "coarse":
                if first:
                    first = False
                    return response(event(1.0, rule_id=definition.id))
                return response()
            item = proof_event(frames, action_sec=1.0)
            item["rule_id"] = "unknown" if invalid == "unknown" else definition.id
            if invalid == "proof":
                item["evidence_timestamps_sec"] = list(reversed(item["evidence_timestamps_sec"]))
            return response(item)

        with pytest.raises(InvalidModelOutputError):
            VideoAnalyzer(RecordingBackend(detect), [rule], observations=[definition],
                config=AnalyzerConfig(sample_fps=2, window_sec=3, overlap_sec=.5, max_edge=64,
                                      refine_max_edge=80)).analyze(video, tmp_path / str(index))


def test_review_is_an_abstention_not_a_negative_prediction(tmp_path, definition):
    sha = "1" * 64
    manifest = {"schema_version": 1, "groups": [{"sha256": sha, "label": 0, "split": "development",
                "filenames": ["synthetic.mov"], "weak_points_sec": []}]}
    result = AnalysisResult(verdict="requires_review", violations=[], observations=[observation(definition)])
    manifest_path, prediction_path = tmp_path / "manifest.json", tmp_path / "predictions.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    prediction_path.write_text(json.dumps([{"sha256": sha, "status": "done",
                                           "result": result.model_dump(mode="json")}]), encoding="utf-8")
    report = evaluate_predictions(manifest_path, prediction_path, split="development")
    assert report["counts"]["processed"] == 0 and report["counts"]["failed"] == 1
    assert report["confusion"]["tn"] == 0
    assert report["metrics"]["accuracy_all"] == 0
