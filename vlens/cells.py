"""Cells: one word position of a text, read at one layer, drawn in one way with one seed. The item files, positions,
record files and picture names shared by scripts/draw_cells.py and the scripts that score its pictures.

Items come from a YAML file in configs/prompts/ with one of three kinds:
- two_hop: {prompt, concept, description_end, ...}. Positions: the last word of the question (the last word before the
  final "?") and the word `description_end`;
- double_meaning: {word, meaning_a, meaning_b, sentence_ab, sentence_ba}. Each sentence is one text; positions: the first
  and the second occurrence of `word` (meaning a then b in sentence_ab, b then a in sentence_ba);
- reads: {id, text, truth, reads: [{name, word, occurrence, truth}]}. Any text with the words to read named explicitly
  (occurrence counts whole words, default -1 = the last; a read's truth defaults to the item's).
"""

import json
import re

import yaml

from vlens import prompts

TARGET = "a photo of a thing"


def word_tokens(reading, word: str, occurrence: int) -> list[int]:
    """Token positions of the given occurrence of `word` as a whole word ("bat" does not match "battle")."""
    starts, text = [], ""
    for s in reading.token_strs:
        starts.append(len(text))
        text += s
    hits = [m.start() for m in re.finditer(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text)]
    if not hits:
        raise ValueError(f"{word!r} not found as a word in {text!r}")
    begin = hits[occurrence]
    end = begin + len(word)
    return [p for p, s in enumerate(starts) if s < end and s + len(reading.token_strs[p]) > begin]


def texts(items_file: str, kind: str, only: list[int] | None) -> list[dict]:
    """One record per text to read: {item, text, positions: [(name, word, occurrence, truth)]}."""
    data = yaml.safe_load((prompts.PROMPTS / f"{items_file}.yaml").read_text())
    out = []
    for i, it in enumerate(data[kind]):
        if only is not None and i not in only:
            continue
        if kind == "two_hop":
            words = re.findall(r"[\w'-]+", it["prompt"])
            last = words[-1]
            out.append(dict(item=i, text=it["prompt"], positions=[
                ("question_end", last, -1, it["concept"]),
                ("description_end", it["description_end"], -1, it["concept"])]))
        elif kind == "reads":
            out.append(dict(item=i, text=it["text"], positions=[
                (r.get("name", f"r{k}"), r["word"], r.get("occurrence", -1), r.get("truth", it.get("truth")))
                for k, r in enumerate(it["reads"])]))
        else:
            a, b = it["meaning_a"], it["meaning_b"]
            name = lambda m: m["name"] if isinstance(m, dict) else m  # noqa: E731
            for order, (first, second) in (("ab", (a, b)), ("ba", (b, a))):
                out.append(dict(item=i, text=it[f"sentence_{order}"], order=order, positions=[
                    ("first", it["word"], 0, name(first)), ("second", it["word"], 1, name(second))]))
    return out


def all_records(folder) -> list[dict]:
    """Every record of a model folder, from all per-job record files, without duplicates."""
    seen, out = set(), []
    for f in sorted(folder.glob("records*.json")):
        for r in json.loads(f.read_text()):
            if r["file"] not in seen:
                seen.add(r["file"])
                out.append(r)
    return out


def picture_name(items, rec, pos_name, layer, way, seed) -> str:
    return f"{items}_{rec['item']:03d}{rec.get('order') or ''}_{pos_name}_L{layer}_{way.replace(':', '-')}_s{seed}.png"
