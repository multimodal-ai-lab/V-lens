"""Table 5: the lens ablation on the test questions of Table 1 (CPU).

One scenario: the share of correctly answered two-step questions whose unnamed object is drawn, per model and averaged
over models. Columns: the V-lens (copy:diag), D = I (the moved copy, way moved), no centering on mu_l
(copy:diag-nocenter), no insertion-layer mean mu_R (copy:diag-nomuR), only sigma_l (copy:diag-srconly), only sigma_R
(copy:diag-dstonly); vlens/carry.py ABLATIONS. Models: BAGEL, Show-o2, SenseNova-U1 (request "thing" under its system
message); HunyuanImage optionally (--models), not in the paper.

Counts are made here from the pictures' records and the three judges' answers with scripts/score_cells.py's rules (a
picture shows the object when 2 of 3 judges choose it, a cell when 2 of 3 seeds do, a question is a hit when any cell
over both positions and all layers is). For every folder that has a scores.json the count is checked against it. A
model enters a set only when every cell of every row is decided (a cell with two agreeing seeds of 0 and 1 is decided)
and every question is counted; otherwise it is "pending". Intervals: 95% paired bootstrap of each ablation's mean rate
minus the V-lens's, 2,000 resamples, each model's questions resampled on their own, seed 20261012.
Unablated rows from fresh2_cells (Test) and heldout_cells / heldout_cells_thing (Val), ablation rows from
lens_ablation_test_cells / lens_ablation_val_cells (scripts/lens_ablation.py shows how they are drawn).

    python scripts/score_ablation.py      -> $VLENS_OUT/ablation/slim_table.{txt,json}
"""

import argparse
import json
import random
from collections import defaultdict

from vlens import cells as dc
from vlens import prompts, results
from vlens.judges import answer_files

R_ = results.RESULTS
OUT = R_ / "ablation"
JUDGES = ["qwen3vl", "gemma3", "llavaov2"]
MODELS = ["bagel", "show_o2", "sensenova"]   # --models: HunyuanImage can be added
TABLE1 = ["bagel", "show_o2", "sensenova"]
NAMES = {"bagel": "BAGEL", "show_o2": "Show-o2", "sensenova": "SenseNova-U1", "hunyuan": "HunyuanImage"}
ROWS = [("V-lens", "copy:diag"), ("D = I (moved copy)", "moved"), ("No centering", "copy:diag-nocenter"),
        ("No insertion-layer mean", "copy:diag-nomuR"), ("Source spreads only", "copy:diag-srconly"),
        ("Insertion spreads only", "copy:diag-dstonly")]
REF = "V-lens"
UNABLATED = ("copy:diag", "moved")   # drawn with the lens-variant cells
ITEMS = {"test": "fresh2_items", "val": "heldout_items"}
# (unablated folder, ablation folder) per set and model; SenseNova-U1 with the request "thing"
SOURCES = {"test": {"bagel": ("fresh2_cells", "lens_ablation_test_cells"),
                    "show_o2": ("fresh2_cells", "lens_ablation_test_cells"),
                    "sensenova": ("fresh2_cells", "lens_ablation_test_cells"),
                    "hunyuan": ("fresh2_cells", "lens_ablation_test_cells")},
           "val": {"bagel": ("heldout_cells", "lens_ablation_val_cells"),
                   "show_o2": ("heldout_cells", "lens_ablation_val_cells"),
                   "sensenova": ("heldout_cells_thing", "lens_ablation_val_cells"),
                   "hunyuan": ("heldout_cells", "lens_ablation_val_cells")}}
ANSWERS = {"test": prompts.REPO / "configs" / "test_sets" / "test.json",
           "val": prompts.REPO / "configs" / "test_sets" / "val.json"}
SEED, N_BOOT = 20261012, 2000

sc = results.load_script("score_cells")   # NEAR / FAR layers per model


def questions(set_, model):
    """The two-step questions the model answers right in text (the units of Table 1 and Table 5); for HunyuanImage the
    test-set file needs its own entry."""
    return sorted(json.loads(ANSWERS[set_].read_text())[model])


_votes = {}


def votes(folder):
    if folder not in _votes:
        v = defaultdict(dict)
        for j in JUDGES:
            for f in answer_files(R_ / "judges", j, f"cells:{folder}"):
                for a in json.loads(f.read_text()):
                    v[a["file"]][j] = a
        _votes[folder] = v
    return _votes[folder]


def count(folder, model, items, way, qs):
    """Hits of one way: {hits, n, undecided, missing_cells, stored} by scripts/score_cells.py's rules, with a decision
    check."""
    base = R_ / folder
    if not (base / model).exists():
        return None
    layers = sc.NEAR[model] + sc.FAR[model]
    v = votes(folder)
    cell = defaultdict(dict)   # (item, position, layer) -> seed -> shows (only pictures all three judges answered)
    drawn = defaultdict(set)
    for r in dc.all_records(base / model):
        if r["kind"] != "two_hop" or r["items"] != items or r["way"] != way or r["layer"] not in layers:
            continue
        key = (r["item"], r["position"], r["layer"])
        drawn[key].add(r["seed"])
        a = v.get(r["file"], {})
        if len(a) == len(JUDGES):
            cell[key][r["seed"]] = sum(x["chosen"] == r["truth"] for x in a.values()) > len(JUDGES) / 2
    if not drawn:
        return None
    positions = sorted({k[1] for k in drawn})
    hits, undecided, missing = set(), set(), 0
    for q in qs:
        hit, open_ = False, False
        for pos in positions:
            for l in layers:
                s = cell.get((q, pos, l), {})
                if (q, pos, l) not in drawn:
                    missing += 1
                    open_ = True
                    continue
                yes = sum(s.values())
                if yes >= 2:
                    hit = True
                elif not (len(s) == 3 or (0 in s and 1 in s and s[0] == s[1]) or (len(s) == 2 and yes == 0 and 2 in s)):
                    open_ = True
        if hit:
            hits.add(q)
        elif open_:
            undecided.add(q)
    units = {k[0] for k in drawn}
    out = dict(hits=hits, n=len(units & set(qs)), undecided=undecided, missing_cells=missing,
               extra_units=sorted(units - set(qs)), positions=positions)
    # the stored scores (scripts/score_cells.py) of this folder, when there are any
    if (base / "scores.json").exists():
        s = json.loads((base / "scores.json").read_text())["scores"].get(model, {}).get("two_hop", {}).get(way, {}).get("all")
        if s is not None:
            out["stored"] = dict(hits=s["hits"], n=s["n"], same_hits={u[0] for u in s["hit_units"]} == hits)
    return out


def build_set(set_):
    rows = {}
    for m in MODELS:
        ref_folder, abl_folder = SOURCES[set_][m]
        qs = questions(set_, m)
        per = {}
        for name, way in ROWS:
            per[name] = count(ref_folder if way in UNABLATED else abl_folder, m, ITEMS[set_], way, qs)
        done = {n: c is not None and c["n"] == len(qs) and not c["undecided"] and not c["missing_cells"]
                for n, c in per.items()}
        rows[m] = dict(questions=qs, rows=per, complete=all(done.values()), row_complete=done)
    return rows


def bootstrap(data, models, rng):
    """Paired bootstrap of every row's mean rate minus the V-lens's, and of each model's own difference."""
    qidx = {m: data[m]["questions"] for m in models}
    hits = {m: {n: data[m]["rows"][n]["hits"] for n, _ in ROWS} for m in models}
    diffs = {n: [] for n, _ in ROWS if n != REF}
    per_model = {m: {n: [] for n in diffs} for m in models}
    for _ in range(N_BOOT):
        sample = {m: [rng.choice(qidx[m]) for _ in qidx[m]] for m in models}
        rate = {m: {n: sum(q in hits[m][n] for q in sample[m]) / len(sample[m]) for n, _ in ROWS} for m in models}
        for n in diffs:
            diffs[n].append(sum(rate[m][n] - rate[m][REF] for m in models) / len(models))
            for m in models:
                per_model[m][n].append(rate[m][n] - rate[m][REF])

    def ci(xs):
        xs = sorted(xs)
        return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]
    return {n: ci(d) for n, d in diffs.items()}, {m: {n: ci(d) for n, d in per_model[m].items()} for m in models}


def summarize(set_, data):
    out = dict(models={}, means={})
    for m in MODELS:
        d = data[m]
        out["models"][m] = dict(complete=d["complete"], n_questions=len(d["questions"]), rows={})
        for name, _ in ROWS:
            c = d["rows"][name]
            out["models"][m]["rows"][name] = None if c is None else dict(
                complete=d["row_complete"][name], hits=len(c["hits"]), n=c["n"], undecided=len(c["undecided"]),
                missing_cells=c["missing_cells"],
                stored=c.get("stored") and dict(hits=c["stored"]["hits"], n=c["stored"]["n"],
                                                same_hits=c["stored"]["same_hits"]))
    groups = (([("three models (Table 5)", TABLE1)] if set(TABLE1) <= set(MODELS) else [])
              + ([("all models", MODELS)] if MODELS != TABLE1 else []))
    for group, models in groups:
        if not all(data[m]["complete"] for m in models):
            out["means"][group] = "pending: " + ", ".join(NAMES[m] for m in models if not data[m]["complete"])
            continue
        rate = {n: sum(len(data[m]["rows"][n]["hits"]) / len(data[m]["questions"]) for m in models) / len(models)
                for n, _ in ROWS}
        ci, per_model = bootstrap(data, models, random.Random(SEED))
        out["means"][group] = dict(
            rate={n: round(r, 4) for n, r in rate.items()},
            minus_vlens={n: round(rate[n] - rate[REF], 4) for n in ci},
            minus_vlens_95=ci, per_model_minus_vlens_95=per_model)
    return out


def fmt_ci(d, c):
    return f"{d:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]"


def text(numbers):
    lines = ["Lens ablation: two-step questions answered right in text whose unnamed object is drawn",
             "(a question is a hit when any of its cells shows the object). Mean = mean of the models' rates;",
             "interval = 95% paired bootstrap over questions.", ""]
    for set_ in ("test", "val"):
        s = numbers[set_]
        lines.append(f"== {set_.capitalize()}")
        head = f"{'':26s}" + "".join(f"{NAMES[m]:>14s}" for m in MODELS)
        means = [g for g, v in s["means"].items() if isinstance(v, dict)]
        head += "".join(f"   {'mean, ' + g.split(' ')[0] + ' models':>16s} {'minus V-lens [95%]':>26s}" for g in means)
        lines.append(head)
        for name, _ in ROWS:
            line = f"{name:26s}"
            for m in MODELS:
                r = s["models"][m]["rows"][name]
                cell = f"{r['hits']}/{r['n']}" if r and r["complete"] else "pending"
                line += f"{cell:>14s}"
            for g in means:
                v = s["means"][g]
                line += f"   {v['rate'][name]:>16.3f} " + (f"{fmt_ci(v['minus_vlens'][name], v['minus_vlens_95'][name]):>26s}"
                                                         if name != REF else f"{'':>26s}")
            lines.append(line)
        for g, v in s["means"].items():
            if not isinstance(v, dict):
                lines.append(f"   mean over {g}: {v}")
        for m in MODELS:
            if s["models"][m]["complete"]:
                continue
            parts = []
            for n, r in s["models"][m]["rows"].items():
                if r is None:
                    parts.append(f"{n} not drawn yet")
                elif not r["complete"]:
                    parts.append(f"{n} {r['hits']} hits so far, {r['undecided']} questions open, "
                                 f"{r['missing_cells']} cells not drawn")
            lines.append(f"   pending {NAMES[m]} (not in any mean): " + "; ".join(parts))
        for g, v in s["means"].items():
            if isinstance(v, dict):
                lines.append(f"   per model, ablation minus V-lens (95% over that model's questions), {g}:")
                for name, _ in ROWS[1:]:
                    lines.append(f"      {name:26s}" + "".join(
                        f"  {NAMES[m]} [{c[name][0]:+.2f}, {c[name][1]:+.2f}]"
                        for m, c in v["per_model_minus_vlens_95"].items()))
        checks = [(m, n, r["stored"]) for m in MODELS for n, r in s["models"][m]["rows"].items() if r and r.get("stored")]
        bad = [(m, n) for m, n, st in checks if not st["same_hits"]]
        lines.append(f"   check against the stored scores.json rows: {len(checks) - len(bad)} of {len(checks)} reproduced"
                     + (f"; DIFFERENT: {bad}" if bad else ""))
        lines.append("")
    return "\n".join(lines)


def main():
    global MODELS
    ap = argparse.ArgumentParser(description="Table 5: the lens ablation on the two-step test questions.")
    ap.add_argument("--models", default=",".join(MODELS), help="comma separated (hunyuan optional, not in the paper)")
    args = ap.parse_args()
    MODELS = args.models.split(",")
    numbers = {}
    for set_ in ("test", "val"):
        numbers[set_] = summarize(set_, build_set(set_))
    OUT.mkdir(parents=True, exist_ok=True)
    results.save_numbers(OUT / "slim_table.json", numbers)
    t = text(numbers)
    (OUT / "slim_table.txt").write_text(t + "\n")
    print(t)
    print("->", OUT / "slim_table.txt")


if __name__ == "__main__":
    main()
