"""Score cell pictures with the judge panel: the lens-variant and ablation tables (Table 1, Table 5, Supp. Tables S2, S3).

Reads the pictures' records (scripts/draw_cells.py) and every judge's answers on them (scripts/run_judge.py
--set cells:<folder>). Per picture: the panel says it shows the concept when most judges pick the concept in the choice
question. Per cell (one word position at one layer, one way of filling the slot): it shows the concept when the panel
says so in at least two of its three seeds. Per question: a hit when at least one of its cells in the group shows it.
Groups: near the insertion layer, far from it (NEAR, FAR), and all layers (the paper's tables use "all"). Reported per
model and way. For words with two meanings the unit is one occurrence of the word, and the main number is the sentences
where both occurrences show their own meaning; "literal" counts units where the cue word's own meaning (two-step) or the
word's other meaning (two meanings) is shown instead. Ways are compared unit by unit (only the first way hits, only the
other way hits) with a one-sided exact sign test.
The control asks the same of another question's concept (the next one that differs from this question's concept), using
the judges' open answers (which name what the picture shows, matched as whole words), since the choice question did not
offer that concept. Writes <folder>/scores.json.

    python scripts/score_cells.py --folder fresh2_cells
"""

import argparse
import json
import math
import re
from collections import defaultdict

from vlens import prompts, results
from vlens.judges import answer_files

JUDGES = ["qwen3vl", "gemma3", "llavaov2"]
NEAR = {"bagel": [6, 8, 10], "show_o2": [20, 24, 27], "sensenova": [20, 24, 28]}
FAR = {"bagel": [2, 4, 14, 20], "show_o2": [6, 10, 14], "sensenova": [8, 12, 16, 36]}
# HunyuanImage (insertion layer 9), optional and not in the paper's variant tables: near 6, 9, 12, far 0, 27, 30
NEAR["hunyuan"] = [6, 9, 12]
FAR["hunyuan"] = [0, 27, 30]


def words_of(name: str) -> list[str]:
    for c in prompts.concepts():
        if c["name"].lower() == name.lower():
            return [w.lower() for w in c["words"]]
    return [name.lower()]


def names(text: str, concept: str) -> bool:
    """Whether `text` names `concept` as a whole word ("ant" is not in "elephant"), plural allowed."""
    return any(re.search(rf"(?<![\w-]){re.escape(w)}(s|es)?(?![\w-])", text.lower()) for w in words_of(concept))


def other_concept(items: dict, item: int, truth: str) -> str:
    """The control concept: the first concept of the following items that differs from `truth` (several development
    items share a concept, so the plain next item would often be the same thing)."""
    order = sorted(items)
    k = order.index(item)
    for step in range(1, len(order) + 1):
        for c in sorted(items[order[(k + step) % len(order)]]):
            if c != truth:
                return c
    raise ValueError("all items share one concept")


def main():
    ap = argparse.ArgumentParser(description="Score the cell pictures of one output folder with the judge panel.")
    ap.add_argument("--folder", required=True, help="output folder of scripts/draw_cells.py, e.g. fresh2_cells")
    args = ap.parse_args()
    base = results.RESULTS / args.folder
    judges_dir = results.RESULTS / "judges"
    votes = defaultdict(dict)  # file -> judge -> answer
    for j in JUDGES:
        for f in answer_files(judges_dir, j, f"cells:{args.folder}"):
            for a in json.loads(f.read_text()):
                votes[a["file"]][j] = a
    records = {}
    for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for rec_file in model_dir.glob("records*.json"):
            for r in json.loads(rec_file.read_text()):
                records[r["file"]] = r
    concepts = {}  # (model, items file) -> item -> its concepts (two for a double-meaning item)
    for r in records.values():
        concepts.setdefault((r["model"], r["items"]), {}).setdefault(r["item"], set()).add(r["truth"])
    shows, shows_next, shows_literal, missing = {}, {}, {}, 0
    for file, r in records.items():
        v = votes.get(file, {})
        if len(v) < len(JUDGES):
            missing += 1
            continue
        nxt = other_concept(concepts[(r["model"], r["items"])], r["item"], r["truth"])
        shows[file] = sum(a["chosen"] == r["truth"] for a in v.values()) > len(JUDGES) / 2
        shows_next[file] = sum(names(a["open"], nxt) for a in v.values()) > len(JUDGES) / 2
        # the literal option: the cue word's own meaning (two-step) or the word's other meaning (two meanings)
        shows_literal[file] = sum(a.get("literal") is not None and a["chosen"] == a["literal"]
                                  for a in v.values()) > len(JUDGES) / 2
    cell = defaultdict(lambda: [[], [], []])  # (model, kind, way, item, order, position, layer) -> votes per seed
    for file, r in records.items():
        if file in shows:
            key = (r["model"], r["kind"], r["way"], r["item"], r.get("order"), r["position"], r["layer"])
            cell[key][0].append(shows[file])
            cell[key][1].append(shows_next[file])
            cell[key][2].append(shows_literal[file])
    # Units: a question (two-step) or one occurrence of the word (two meanings, keyed by item, order and position).
    table = {}
    for (model, kind, way, item, order, position, layer), (yes, ctl, lit) in cell.items():
        unit = (item,) if kind == "two_hop" else (item, order, position)
        for group, layers in (("near", NEAR[model]), ("far", FAR[model]), ("all", NEAR[model] + FAR[model])):
            if layer not in layers and way != "word":
                continue
            t = table.setdefault((model, kind, way, group), dict(units=set(), hits=set(), control=set(), literal=set()))
            t["units"].add(unit)
            for votes_, key in ((yes, "hits"), (ctl, "control"), (lit, "literal")):
                if sum(votes_) >= 2:
                    t[key].add(unit)
    out = {}
    for (model, kind, way, group), t in sorted(table.items()):
        row = dict(n=len(t["units"]), hits=len(t["hits"]), control=len(t["control"]), literal=len(t["literal"]),
                   hit_units=sorted(t["hits"], key=str))
        if kind == "double_meaning":  # sentences where both occurrences show their own meaning
            row["both"] = len({(i, o) for i, o, _ in t["hits"]
                               if all((i, o, q) in t["hits"] for q in ("first", "second"))})
            row["sentences"] = len({(i, o) for i, o, _ in t["units"]})
        out.setdefault(model, {}).setdefault(kind, {}).setdefault(way, {})[group] = row
    # Ways compared unit by unit: only the first way hits vs only the other way hits; one-sided exact sign test.
    compare = {}
    for (model, kind, way, group), t in table.items():
        if way == "word":
            continue
        # every Jacobian-map way ("vlens...") against the copies (and a map variant against the model's standard map, the
        # vlens way with the shortest name); every way against the word on its own (does the context add the concept?)
        vlens_ways = sorted((w for (m_, k_, w, g_) in table if (m_, k_, g_) == (model, kind, group) and w.startswith("vlens")),
                            key=len)
        standard = [vlens_ways[0]] if way.startswith("vlens") and vlens_ways and way != vlens_ways[0] else []
        if way.startswith("copy:"):  # a copy without a map against the moved copy, the plain copy, the word and the map
            standard = vlens_ways[:1]
        others = (("copy", "moved", "word", *standard) if way.startswith(("vlens", "copy:")) else ("word",))
        for other in others:
            o = table.get((model, kind, other, group))
            if o is None:
                continue
            only_v, only_o = len(t["hits"] - o["hits"]), len(o["hits"] - t["hits"])
            n = only_v + only_o
            p = sum(math.comb(n, k) for k in range(only_v, n + 1)) / 2 ** n if n else 1.0
            compare.setdefault(model, {}).setdefault(kind, {}).setdefault(f"{way} vs {other}", {})[group] = dict(
                only_first=only_v, only_other=only_o, p_one_sided=round(p, 4))
    for model, kinds in out.items():
        for kind, ways in kinds.items():
            print(f"== {model} {kind}")
            for way, groups in ways.items():
                print(f"   {way:58s} " + "   ".join(
                    f"{g} {v['hits']:2d}/{v['n']}" + (f" both {v['both']}/{v['sentences']}" if "both" in v else "")
                    + f" (control {v['control']}, literal {v['literal']})" for g, v in groups.items()))
            for pair, groups in compare.get(model, {}).get(kind, {}).items():
                print(f"   {pair:58s} " + "   ".join(f"{g} {v['only_first']}:{v['only_other']} p={v['p_one_sided']}"
                                                   for g, v in groups.items()))
    print(f"pictures without all three judges yet: {missing}")
    results.save_numbers(base / "scores.json", dict(scores=out, compare=compare, missing=missing))


if __name__ == "__main__":
    main()
