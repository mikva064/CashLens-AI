"""Visible-event catalog is separate from legally qualified rules."""
import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from .contracts import ObservationDefinition, Rule
from .errors import RuleValidationError

PIPELINE_REVISION = "observable-catalog-v3"
_RULE_DESCRIPTIONS = {
    "cashier_leaves_during_service": "Во время продолжающегося приёма наличных работник покидает своё рабочее место.",
    "prior_receipt_cash_exposed": "При следующем приёме наличных ранее принятая партия остаётся открыто на рабочей поверхности.",
    "receipt_count_without_client": "Пересчёт прослеженной партии принимаемых денег продолжается после ухода передавшего их посетителя.",
}


def category_description(definition: Rule | ObservationDefinition) -> str:
    if isinstance(definition, ObservationDefinition):
        return definition.description_template
    return _RULE_DESCRIPTIONS.get(definition.id, definition.title)


def load_observations(path: str | Path) -> list[ObservationDefinition]:
    try:
        definitions = TypeAdapter(list[ObservationDefinition]).validate_python(
            json.loads(Path(path).read_text(encoding="utf-8-sig")))
    except (OSError, ValueError) as exc:
        raise RuleValidationError("Cannot read a valid visual observation catalog") from exc
    if len({definition.id for definition in definitions}) != len(definitions):
        raise RuleValidationError("Duplicate visual observation IDs")
    return definitions


def catalog_hash(rules: list[Rule], observations: list[ObservationDefinition]) -> str:
    data = {"pipeline_revision": PIPELINE_REVISION,
            "rules": [{**rule.model_dump(mode="json"), "description_template": category_description(rule)}
                      for rule in sorted(rules, key=lambda item: item.id)],
            "observations": [item.model_dump(mode="json")
                             for item in sorted(observations, key=lambda item: item.id)]}
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
