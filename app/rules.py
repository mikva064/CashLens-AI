"""Справочник правил: текст нормы и ссылка берутся отсюда, а не из модели."""

import json
from functools import lru_cache

from app.config import RULES_PATH


@lru_cache(maxsize=1)
def load() -> dict:
    with open(RULES_PATH, encoding="utf-8") as f:
        return json.load(f)["types"]


def lookup(event_type: str) -> dict:
    rule = load().get(event_type)
    if rule is None:
        return {"id": event_type, "title": event_type, "requirement": "",
                "source": "Тип отсутствует в справочнике", "url": "", "verified": False}
    return {"id": event_type, **rule}
