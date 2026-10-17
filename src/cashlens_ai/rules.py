"""Load human-reviewed rules; sources are never invented by the model."""

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .contracts import Rule
from .errors import RuleValidationError


def load_rules(path: str | Path) -> list[Rule]:
    try:
        rules = TypeAdapter(list[Rule]).validate_python(
            json.loads(Path(path).read_text(encoding="utf-8-sig"))
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise RuleValidationError("Cannot read a valid rules registry") from exc
    if not rules:
        raise RuleValidationError(
            "Rules registry is empty. Review the local videos, define observable "
            "event types and verify their sources before running classification."
        )
    if len({r.id for r in rules}) != len(rules):
        raise RuleValidationError("Duplicate rule IDs")
    if any(not r.verified for r in rules):
        raise RuleValidationError("Every active rule must be manually verified")
    return rules
