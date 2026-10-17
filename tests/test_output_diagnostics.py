"""Fixed diagnostics distinguish invalid proof without exposing model content."""
import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from app import worker
from cashlens_ai.backend import _expand_wire_response
from cashlens_ai.errors import InvalidModelOutputError
from cashlens_ai.pipeline import parse_model_response


RULES = [{"id": "test_rule", "title": "Synthetic category",
          "observable_condition": "A complete synthetic physical sequence is visible."}]
PROOF = {"k": "test_rule", "b": 0, "f": 2, "a": 4, "c": 2}


class DiagnosticAnalyzer:
    name = "model"

    def load(self):
        logging.getLogger("cashlens.model").info("synthetic_generation nframes=0 generated_tokens=0")

    def capabilities(self):
        return {}

    def close(self):
        pass


def diagnostic_factory():
    return DiagnosticAnalyzer()


@pytest.mark.parametrize("raw,phase,expected", [
    ('{"e":[PRIVATE_SYNTHETIC_TEXT', "coarse", "invalid_compact_json"),
    ('{"e":[],"e":[]}', "coarse", "duplicate_json_key"),
    ('{"e":[{"k":"test_rule","f":NaN}]}', "coarse", "nonfinite_json"),
    ('{"events":[]}', "coarse", "invalid_compact_shape"),
    (json.dumps({"e": [{"k": "test_rule", "f": 2}] * 4}), "coarse", "too_many_events"),
    ('{"e":[{"k":"test_rule"}]}', "coarse", "invalid_event_fields"),
    ('{"e":[{"k":"unknown","f":2}]}', "coarse", "unknown_category"),
    ('{"e":[{"k":"test_rule","f":true}]}', "coarse", "invalid_action_index"),
    (json.dumps({"e": [{"k": "test_rule", "f": 2}] * 2}), "coarse", "duplicate_event"),
    (json.dumps({"e": [{**PROOF, "b": False}]}), "fine", "invalid_proof_index"),
    (json.dumps({"e": [{**PROOF, "b": 3}]}), "fine", "invalid_proof_order"),
    (json.dumps({"e": [{**PROOF, "c": 1}]}), "fine", "invalid_cash_context"),
    (json.dumps({"e": [{**PROOF, "f": 3, "c": 3}]}), "fine", "candidate_focus"),
])
def test_strict_wire_failure_has_fixed_reason(raw, phase, expected):
    focus = {"candidate_timecode_sec": 2.0, "candidate_radius_sec": 0.1} if phase == "fine" else {}
    with pytest.raises(InvalidModelOutputError) as caught:
        _expand_wire_response(raw, RULES, [0., 1., 2., 3., 4.], phase, **focus)
    assert caught.value.reason_code == expected
    assert "PRIVATE_SYNTHETIC_TEXT" not in str(caught.value)


def test_reason_codes_are_allowlisted_even_after_external_attribute_mutation():
    secret = "PRIVATE_SYNTHETIC_REASON"
    error = InvalidModelOutputError(secret, reason_code=secret)
    assert error.reason_code == "invalid_model_output"
    error.reason_code = secret
    message = worker._error_message(error, "synthetic-job")
    assert message["reason_code"] == "invalid_model_output"
    assert secret not in json.dumps(message)
    assert worker._error_message(RuntimeError(secret))["reason_code"] == "invalid_model_output"


def test_pipeline_proof_failure_and_parse_failure_keep_separate_reasons():
    event = {"rule_id": "test_rule", "timecode_sec": 2, "description": "Synthetic sequence",
             "evidence_timestamps_sec": [0., 2., 1.], "cash_context_timestamp_sec": 0.}
    with pytest.raises(InvalidModelOutputError) as caught:
        parse_model_response(json.dumps({"events": [event]}), allowed_ids={"test_rule"},
                             first_sec=0, last_sec=4, require_proof=True,
                             frame_timestamps=[0., 1., 2., 3., 4.],
                             candidate_timecode_sec=2, candidate_radius_sec=0.1)
    assert caught.value.reason_code == "invalid_source_proof"
    with pytest.raises(InvalidModelOutputError) as caught:
        parse_model_response('{"events":[', allowed_ids={"test_rule"}, first_sec=0, last_sec=4)
    assert caught.value.reason_code == "invalid_event_json"


@pytest.mark.parametrize("supplied_code,expected", [("invalid_proof_order", "invalid_proof_order"),
                                                    ("PRIVATE_SYNTHETIC_REASON", "invalid_model_output")])
def test_parent_log_contains_only_safe_failure_metadata(monkeypatch, caplog, supplied_code, expected):
    message = {"kind": "failed", "job_id": "synthetic-job", "error_type": "InvalidModelOutputError",
               "reason_code": supplied_code, "error": "PRIVATE_SYNTHETIC_CONTROL_TEXT"}
    runner = SimpleNamespace(send=lambda _: None, poll=lambda _: message, is_alive=lambda: True)
    monkeypatch.setattr(worker.db, "get", lambda _: {"created_at": time.time()})
    monkeypatch.setattr(worker.db, "fail", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_expire_queue", lambda: None)
    with caplog.at_level(logging.ERROR, logger="cashlens.worker"):
        assert worker._process_job(threading.Event(), runner, "synthetic-job") is True
    assert f"reason_code={expected}" in caplog.text
    assert "PRIVATE_SYNTHETIC" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_spawned_child_emits_info_metadata_with_warning_parent(monkeypatch, capfd):
    monkeypatch.setattr(logging.getLogger(), "level", logging.WARNING)
    runner = worker.ModelProcess(factory=diagnostic_factory)
    try:
        runner.start()
        until = time.monotonic() + 8
        message = None
        while time.monotonic() < until and message is None:
            message = runner.poll(0.05)
        assert message and message["kind"] == "ready"
    finally:
        runner.close()
    assert "synthetic_generation nframes=0 generated_tokens=0" in capfd.readouterr().err
