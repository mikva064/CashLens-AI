"""Review aggregation uses only synthetic records and never video files."""

import json
import time

import pytest

def feedback_counts(rule_ids):
    from app.feedback_stats import feedback_counts as aggregate
    return aggregate(rule_ids)


def set_feedback(*args):
    from app import db
    db.set_feedback(*args)


@pytest.fixture
def database(monkeypatch, tmp_path):
    from app import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "feedback.sqlite3")
    db.init()


def valid_result(*rule_ids):
    return {"analysis": {"schema_version": "1.0", "verdict": "violation" if rule_ids else "no_violation",
                         "violations": [{
                             "rule_id": rule_id, "type": "Synthetic event", "timecode_sec": index + 1,
                             "description": "SYNTHETIC_PRIVATE_DESCRIPTION",
                             "source": {"document": "Test source", "section": "1",
                                        "url": "https://example.invalid/rule"},
                             "evidence_frames": [{"timestamp_sec": index + 1,
                                                  "image_path": f"synthetic_{index}.jpg"}],
                         } for index, rule_id in enumerate(rule_ids)]}}


def store_job(job_id, result, *, status="done", raw=False):
    from app import db
    with db._conn() as connection:
        connection.execute(
            "INSERT INTO jobs (id,status,filename,created_at,result) VALUES (?,?,?,?,?)",
            (job_id, status, "synthetic.mov", time.time(), result if raw else json.dumps(result)),
        )


def test_current_upsert_decision_is_counted_once(database):
    store_job("one", valid_result("rule_one"))
    set_feedback("one", 0, "confirmed", "SYNTHETIC_PRIVATE_REVIEW")
    assert feedback_counts(["rule_one"])["by_rule"] == [
        {"rule_id": "rule_one", "confirmed": 1, "rejected": 0}]
    set_feedback("one", 0, "rejected", "replacement")
    assert feedback_counts(["rule_one"])["by_rule"] == [
        {"rule_id": "rule_one", "confirmed": 0, "rejected": 1}]


def test_distinct_rules_events_jobs_and_historical_ids(database):
    store_job("one", valid_result("rule_a", "rule_b", "historical_z"))
    store_job("two", valid_result("rule_b", "historical_c"))
    for job_id, index, decision in (("one", 0, "confirmed"), ("one", 1, "rejected"),
                                    ("one", 2, "confirmed"), ("two", 0, "confirmed"),
                                    ("two", 1, "rejected")):
        set_feedback(job_id, index, decision, "private comment")
    assert feedback_counts(["rule_b", "rule_a", "zero_rule", "rule_b"]) == {"by_rule": [
        {"rule_id": "rule_b", "confirmed": 1, "rejected": 1},
        {"rule_id": "rule_a", "confirmed": 1, "rejected": 0},
        {"rule_id": "zero_rule", "confirmed": 0, "rejected": 0},
        {"rule_id": "historical_c", "confirmed": 0, "rejected": 1},
        {"rule_id": "historical_z", "confirmed": 1, "rejected": 0},
    ]}


def test_active_rules_have_zero_counts_without_reviews(database):
    assert feedback_counts(["first", "second"]) == {"by_rule": [
        {"rule_id": "first", "confirmed": 0, "rejected": 0},
        {"rule_id": "second", "confirmed": 0, "rejected": 0}]}
    assert feedback_counts([]) == {"by_rule": []}


def test_orphans_unfinished_jobs_and_invalid_event_indices_are_ignored(database):
    store_job("done", valid_result("rule_a"))
    store_job("failed", valid_result("rule_a"), status="failed")
    store_job("processing", valid_result("rule_a"), status="processing")
    store_job("queued", valid_result("rule_a"), status="queued")
    for job_id, index in (("orphan", 0), ("failed", 0), ("processing", 0),
                          ("queued", 0), ("done", -1), ("done", 5)):
        set_feedback(job_id, index, "confirmed", None)
    assert feedback_counts(["rule_a"])["by_rule"] == [
        {"rule_id": "rule_a", "confirmed": 0, "rejected": 0}]


@pytest.mark.parametrize("raw", ["not JSON", "null", "[]", "{}", '{"analysis":null}',
                                   '{"analysis":{"violations":[]}}',
                                   '{"analysis":{"verdict":"violation","violations":[]}}'])
def test_malformed_legacy_results_are_ignored(database, raw):
    store_job("legacy", raw, raw=True)
    set_feedback("legacy", 0, "confirmed", "private")
    assert feedback_counts([]) == {"by_rule": []}


def test_invalid_decision_empty_rule_id_and_no_event_result_are_ignored(database):
    store_job("unknown_decision", valid_result("rule_a"))
    set_feedback("unknown_decision", 0, "invalid_legacy_decision", None)
    store_job("empty_id", valid_result(" "))
    set_feedback("empty_id", 0, "confirmed", None)
    store_job("clean", valid_result())
    set_feedback("clean", 0, "confirmed", None)
    assert feedback_counts([]) == {"by_rule": []}


def test_response_contains_only_rule_ids_and_numeric_counts(database):
    store_job("one", valid_result("rule_one"))
    set_feedback("one", 0, "confirmed", "SYNTHETIC_PRIVATE_REVIEW")
    result = feedback_counts(["rule_one"])
    text = json.dumps(result)
    assert "SYNTHETIC_PRIVATE" not in text
    assert set(result) == {"by_rule"}
    assert set(result["by_rule"][0]) == {"rule_id", "confirmed", "rejected"}
