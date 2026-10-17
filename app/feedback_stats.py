"""Aggregate human decisions without returning scenes or reviewer comments."""

import json
from collections.abc import Iterable
from contextlib import closing

from pydantic import ValidationError

from app import db
from cashlens_ai.contracts import AnalysisResult


def feedback_counts(rule_ids: Iterable[str]) -> dict:
    """Count the current upserted decision for valid events in completed jobs.

    Registry IDs retain their input order, including zero-count rules. Valid
    historical IDs absent from that registry follow in deterministic order.
    Missing jobs, malformed legacy results and stale event indices contribute
    nothing; no descriptions or comments enter the returned aggregate.
    """
    active = list(dict.fromkeys(rule_id for rule_id in rule_ids
                                if isinstance(rule_id, str) and rule_id.strip()))
    counts = {rule_id: {"rule_id": rule_id, "confirmed": 0, "rejected": 0}
              for rule_id in active}
    parsed_jobs: dict[str, list[str] | None] = {}
    with closing(db._conn()) as connection:
        rows = connection.execute(
            """SELECT f.job_id, f.event_index, f.decision, j.result
               FROM feedback AS f JOIN jobs AS j ON j.id=f.job_id
               WHERE j.status='done'"""
        ).fetchall()

    for row in rows:
        decision = row["decision"]
        index = row["event_index"]
        if decision not in {"confirmed", "rejected"} or type(index) is not int or index < 0:
            continue
        job_id = row["job_id"]
        if job_id not in parsed_jobs:
            try:
                stored = json.loads(row["result"])
                analysis = AnalysisResult.model_validate(stored["analysis"])
                ids = [event.rule_id for event in analysis.violations]
                parsed_jobs[job_id] = ids if all(rule_id.strip() for rule_id in ids) else None
            except (TypeError, ValueError, KeyError, ValidationError, RecursionError):
                parsed_jobs[job_id] = None
        ids = parsed_jobs[job_id]
        if ids is None or index >= len(ids):
            continue
        rule_id = ids[index]
        counts.setdefault(rule_id, {"rule_id": rule_id, "confirmed": 0, "rejected": 0})[decision] += 1

    historical = sorted(set(counts).difference(active))
    return {"by_rule": [counts[rule_id] for rule_id in [*active, *historical]]}
