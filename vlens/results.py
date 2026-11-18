"""Where outputs go: one folder per experiment, $VLENS_OUT/<name>/, holding pictures, records and numbers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from vlens.paths import OUT, REPO

RESULTS = OUT


def folder(name: str, date: str | None = None) -> Path:
    """The output folder of one experiment, created if missing (`date` is accepted and ignored)."""
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_numbers(path: Path, numbers: dict) -> None:
    path.write_text(json.dumps(numbers, indent=2, default=float))


def load_script(name: str):
    """Import another file of scripts/ as a module (several scripts share helpers this way)."""
    path = REPO / "scripts" / (name if name.endswith(".py") else f"{name}.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
