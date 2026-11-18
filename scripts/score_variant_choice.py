"""The choice of the lens variant on the validation set, by a rule written before drawing (Supp. A.3, Table S2), and
the comparison with the moved copy on the test set (CPU).

Reads the scores of scripts/score_cells.py (Val: BAGEL and Show-o2 in heldout_cells, SenseNova-U1 in
heldout_cells_thing) and, per condition, the mean of six rates: each model's two-step hit rate and its rate of sentences
with both meanings (all layers). The uncertainty of each condition's score minus the best one comes from a paired
bootstrap over two-step questions and double-meaning words (both sentence orders of a word kept together; 2,000
resamples, seed 20261012). Chosen: the simplest condition within one bootstrap standard error of the best, in the
order moved copy, relative size, percentile size, coordinate scaling (the V-lens). Also printed: every condition minus
the moved copy with its 95% interval (on Test: python scripts/score_variant_choice.py --folder fresh2_cells
--sets configs/test_sets/test.json --out <file>).

    python scripts/score_variant_choice.py
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from vlens import prompts, results
from vlens.models import transport_name

MODELS = ["bagel", "show_o2", "sensenova"]
FOLDER = {"bagel": "heldout_cells", "show_o2": "heldout_cells", "sensenova": "heldout_cells_thing"}
CHOICES = [("moved copy", "moved"), ("relative size", "copy:rel"), ("percentile size", "copy:pct"),
           ("coordinate scaling", "copy:diag")]   # coordinate scaling = the V-lens
REFERENCE = [("plain copy", "copy"), ("word on its own", "word")]
SEED, N_BOOT = 20261012, 2000


def main():
    ap = argparse.ArgumentParser(description="Choose the lens variant on Val by the one-standard-error rule.")
    ap.add_argument("--folder", default="", help="one output folder for all models (default: the Val folders)")
    ap.add_argument("--sets", default=str(prompts.REPO / "configs" / "test_sets" / "val.json"),
                    help="the two-step questions each model answers (configs/test_sets/val.json or test.json)")
    ap.add_argument("--out", default=str(results.RESULTS / "copy_transforms" / "choice.json"))
    args = ap.parse_args()
    if args.folder:
        for m in MODELS:
            FOLDER[m] = args.folder
    data, units = {}, {}
    for m in MODELS:
        s = json.loads((results.RESULTS / FOLDER[m] / "scores.json").read_text())["scores"][m]
        std = transport_name(m)
        ways = dict(CHOICES + REFERENCE + [("Jacobian map", f"vlens:{std}"),
                                           ("Jacobian map with gain", f"vlens:{std}_selfgain")])
        data[m] = {}
        for name, way in ways.items():
            th, dm = s["two_hop"].get(way, {}).get("all"), s["double_meaning"].get(way, {}).get("all")
            if th and dm:
                occ = {tuple(u) for u in dm["hit_units"]}
                both = {(i, o) for i, o, _ in occ if all((i, o, q) in occ for q in ("first", "second"))}
                data[m][name] = ({u[0] for u in th["hit_units"]}, both)
        units[m] = (s["two_hop"]["moved"]["all"]["n"], s["double_meaning"]["moved"]["all"]["sentences"] // 2)
    names = [n for n, _ in CHOICES + REFERENCE] + ["Jacobian map", "Jacobian map with gain"]
    names = [n for n in names if all(n in data[m] for m in MODELS)]

    def score(name, q_sample=None, w_sample=None):
        rates = []
        for m in MODELS:
            hits_q, both = data[m][name]
            n_q, n_w = units[m]
            if q_sample is None:
                rates.append(len(hits_q) / n_q)
                rates.append(len(both) / (2 * n_w))
            else:
                rates.append(sum(q in hits_q for q in q_sample[m]) / len(q_sample[m]))
                rates.append(sum(((w, "ab") in both) + ((w, "ba") in both) for w in w_sample[m]) / (2 * len(w_sample[m])))
        return statistics.mean(rates)

    full = {n: score(n) for n in names}
    # population of units for the bootstrap: all test questions (hit or not) and all words
    tests = json.loads(Path(args.sets).read_text())
    q_pop = {m: sorted(tests[m]) for m in MODELS}
    w_pop = {m: list(range(units[m][1])) for m in MODELS}
    rng = random.Random(SEED)
    boots = {n: [] for n in names}
    for _ in range(N_BOOT):
        qs = {m: [rng.choice(q_pop[m]) for _ in q_pop[m]] for m in MODELS}
        ws = {m: [rng.choice(w_pop[m]) for _ in w_pop[m]] for m in MODELS}
        for n in names:
            boots[n].append(score(n, qs, ws))
    best = max((n for n, _ in CHOICES if n in full), key=full.get)
    out = dict(scores=full, best=best, diff_se={})
    for n in names:
        d = [a - b for a, b in zip(boots[n], boots[best])]
        out["diff_se"][n] = statistics.pstdev(d)
        print(f"{n:20s} score {full[n]:.4f}  minus best {full[n] - full[best]:+.4f}  bootstrap SE {out['diff_se'][n]:.4f}")
    chosen = next(n for n, _ in CHOICES if n in full and full[best] - full[n] <= out["diff_se"][n])
    out["chosen"] = chosen
    print("best:", best, "| chosen by the rule (simplest within one SE):", chosen)
    # 95% intervals of each condition minus the moved copy (the comparison reported on the test set)
    out["minus_moved_95"] = {}
    for n in names:
        d = sorted(a - b for a, b in zip(boots[n], boots["moved copy"]))
        out["minus_moved_95"][n] = (d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1])
        print(f"{n:20s} minus moved copy: {full[n] - full['moved copy']:+.4f}, 95% interval "
              f"[{out['minus_moved_95'][n][0]:+.4f}, {out['minus_moved_95'][n][1]:+.4f}]")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.save_numbers(Path(args.out), out)


if __name__ == "__main__":
    main()
