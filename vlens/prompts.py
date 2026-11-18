"""Loading the item files in configs/prompts/."""

from __future__ import annotations

from functools import lru_cache

import yaml

from vlens.paths import REPO

PROMPTS = REPO / "configs" / "prompts"


@lru_cache
def concepts() -> list[dict]:
    """All 100 concepts: [{name, words, group}], in file order."""
    data = yaml.safe_load((PROMPTS / "concepts.yaml").read_text())
    return [dict(item, group=group) for group, items in data.items() for item in items]


def concept_names() -> list[str]:
    return [c["name"] for c in concepts()]


def concept_index(name: str) -> int:
    return concept_names().index(name)


@lru_cache
def _ladder() -> dict:
    return yaml.safe_load((PROMPTS / "ladder.yaml").read_text())


def ladder(level: str) -> list[dict]:
    """Items of one level of configs/prompts/ladder.yaml: [{level, concept, prompt, phrasing, answer?}]."""
    spec = _ladder()[level]
    if "items" in spec:
        return [dict(item, level=level, phrasing=0) for item in spec["items"]]
    names = spec.get("concepts", concept_names())
    return [dict(level=level, concept=name, prompt=t.format(concept=name), phrasing=i)
            for name in names for i, t in enumerate(spec["templates"])]
