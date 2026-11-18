"""Score the text lenses on the context-test items like the pictures (Supp. D, the context columns of the text-readout
table). CPU only; reads text_lenses_<model>.json from scripts/text_lens_ranks.py.

Two criteria per text. "top10" (the table's): the lens ranks the text's own name among its ten most likely tokens and
above the other thing's name. "prefers" (a weaker two-way check): it ranks its own name above the other thing's. A pair
(A descriptions, B pronouns) counts when both of its texts meet the criterion at the same layer; C (two meanings): a
sentence when both occurrences do, a word's score the mean over its two sentences. Only units whose two names start with
different tokens are scored (29 of the 30 description pairs). Reported at the insertion layer (the table) and, for
"prefers", also as the mean over the three fixed layers, with the number of scored units.
Writes $VLENS_OUT/followup_cells/text_lens_scores.json.

    python scripts/score_text_lenses.py
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import yaml

from vlens import prompts, results

FOLDER = results.RESULTS / "followup_cells"
INSERT = {"bagel": 8, "show_o2": 27, "sensenova": 24, "hunyuan": 9}


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    meta = {n: yaml.safe_load((prompts.PROMPTS / f"{n}.yaml").read_text())["reads"]
            for n in ("followup_matched", "followup_reference")}
    out = {}
    for model in INSERT:
        f = FOLDER / f"text_lenses_{model}.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        lenses = sorted({k[:-5] for k in rows[0] if k.endswith("_rank")})
        for lens in lenses:
            for fam, items in (("A", "followup_matched"), ("B", "followup_reference"), ("C", "followup_meanings")):
                cells = defaultdict(lambda: defaultdict(dict))  # unit -> layer -> key -> (prefers own, top10, distinct)
                for r in rows:
                    if r["items"] != items:
                        continue
                    if fam == "C":
                        unit, key = r["item"], (r["order"], r["position"])
                    else:
                        it = meta[items][r["item"]]
                        if it["role"] not in ("text1", "text2"):
                            continue
                        unit, key = it["pair"], it["role"]
                    own, other = r[f"{lens}_rank"], r[f"{lens}_rank_other"]
                    cells[unit][r["layer"]][key] = (own < other, own <= 10 and own < other, r["distinct"])
                need = 4 if fam == "C" else 2
                for crit, c in (("prefers", 0), ("top10", 1)):
                    scores = defaultdict(list)
                    for unit, by_layer in cells.items():
                        if not all(x[2] for v in by_layer.values() for x in v.values()):
                            continue
                        for layer, v in by_layer.items():
                            if len(v) < need:
                                continue
                            if fam == "C":
                                s = np.mean([v[(o, "first")][c] and v[(o, "second")][c] for o in ("ab", "ba")])
                            else:
                                s = float(all(x[c] for x in v.values()))
                            scores[layer].append(s)
                    if not scores:
                        continue
                    ins = scores.get(INSERT[model], [])
                    key = f"{model}|{lens}|{fam}" + ("" if crit == "prefers" else "|top10")
                    out[key] = dict(n_units=len(ins), insertion=float(np.mean(ins)) if ins else None,
                                    insertion_count=float(np.sum(ins)) if ins else None,
                                    band=float(np.mean([np.mean(v) for v in scores.values()])))
    (FOLDER / "text_lens_scores.json").write_text(json.dumps(out, indent=1))
    for k, v in sorted(out.items()):
        print(f"{k:24s} n={v['n_units']:2d} insertion {v['insertion']:.2f} ({v['insertion_count']:g}) band {v['band']:.2f}")


if __name__ == "__main__":
    main()
