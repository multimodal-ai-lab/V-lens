"""Lens ablation (Table 5): the V-lens with one component changed at a time, and the CPU checks behind it.

The V-lens (way copy:diag of scripts/draw_cells.py) carries a state h from its layer l to the insertion layer R as
v = mu_R + s_R D x / |D x|, x = h - mu_l, D = diag(sigma_R / sigma_l). Each ablation is the way copy:diag-<ablation>
(vlens/carry.py, ABLATIONS): nocenter (no mu_l), nomuR (no mu_R), srconly (only sigma_l), dstonly (only sigma_R), and
further ones not in Table 5 (nosize, frac8). D = I is the moved copy (way moved). They are drawn with the same items,
positions, layers, requests, seeds and sampler as the lens-variant cells, judged by the same three judges
(scripts/run_judge.py, choice question) and scored by scripts/score_cells.py; scripts/score_ablation.py makes Table 5.

CPU steps, no model:
- stats: the statistics of the ablations that change only the statistics (variance floor 0 and 10%, an eighth of the
  passages), stored as LENS_STORE/<model>_copy_stats_<ablation>/stats.pt (vlens.carry.build_ablation). Needed before
  drawing the frac8 ablation.
- check: on the saved held-out passage states, every ablation against the V-lens (how far it moves the inserted vector),
  and the switch-off checks (nothing ablated reproduces copy:diag bit for bit; the standard statistics are rebuilt
  exactly; the scalar rescaling is the moved copy). -> lens_ablation/check.json
- table: per model two-step questions whose object is drawn and sentences with both meanings, the six-rate score of
  the three models and the paired bootstrap interval of every row minus the V-lens (the code, seed and order of sets of
  scripts/score_variants.py). Unablated rows from fresh2_cells (Test) and heldout_cells / heldout_cells_thing (Val);
  ablation rows from lens_ablation_test_cells / lens_ablation_val_cells. -> lens_ablation/table.json

Drawing (one call per model and kind; Test shown; Val uses heldout_items, configs/test_sets/val.json and
lens_ablation_val_cells):
    python scripts/draw_cells.py --model bagel --items fresh2_items --kind two_hop --test-set test \\
        --layers 2,4,6,8,10,14,20 --seeds 0,1,2 --folder lens_ablation_test_cells \\
        --ways copy:diag-nocenter,copy:diag-nomuR,copy:diag-nosize,copy:diag-srconly,copy:diag-dstonly,copy:diag-frac8

    python scripts/lens_ablation.py stats
    python scripts/lens_ablation.py check
    python scripts/lens_ablation.py table
"""

import argparse
import json
import random
import statistics

import torch

from vlens import carry as ct
from vlens import paths, results

R_ = results.RESULTS
OUT = R_ / "lens_ablation"
MAIN = ["bagel", "show_o2", "sensenova"]
LAYERS = {"bagel": [2, 4, 6, 8, 10, 14, 20], "show_o2": [6, 10, 14, 20, 24, 27], "sensenova": [8, 12, 16, 20, 24, 28, 36]}
# the ablations that are drawn; scalar, floor0 and floor10 are not (check.json: identical to the moved copy or the V-lens)
DRAWN = ["nocenter", "nomuR", "nosize", "srconly", "dstonly", "frac8"]
WAYS = [f"copy:diag-{a}" for a in DRAWN]


# --- statistics and the CPU check ---------------------------------------------------------------------------------
def stats():
    for m in MAIN:
        for folder in ct.build_ablation(m):
            s = torch.load(folder / "stats.pt", weights_only=False)
            print(m, folder.name, "floor", s["floor"], "fraction", s["fraction"], "tokens", s["tokens"],
                  "s_R", round(s["spread"], 3))


def check(n_tokens: int = 400):
    from vlens.carry import ABLATIONS, ALT_STATS, CopyTransform, ablation_stats, diag_carry
    out = {}
    for m in MAIN:
        c = CopyTransform(m)
        s, R = c.s, c.R
        mu_R, sp = s["mean"][R], s["spread"]
        # 1. the standard statistics rebuilt by ablation_stats are the stored ones, exactly
        std = ablation_stats(m)
        rebuilt = dict(mean=torch.equal(std["mean"], s["mean"].float()), spread=std["spread"] == s["spread"],
                       sigma=all(torch.equal(std["sigma"][l], s["sigma"][l]) for l in s["layers"]))
        # 2. where the variance floors bind: coordinates whose squared spread is below 1% / 10% of the layer's median
        floors = {}
        for l in sorted(set(LAYERS[m]) | {R}):
            f0, f10 = c.stats("floor0")["sigma"][l], c.stats("floor10")["sigma"][l]
            var = f0.pow(2)
            floors[l] = dict(below_1pct=int((var < 0.01 * var.median()).sum()), below_10pct=int((var < 0.1 * var.median()).sum()),
                             min_over_median=round(float(var.min() / var.median()), 4),
                             sigma_floor0_equals_standard=torch.equal(f0, s["sigma"][l]),
                             sigma_floor10_coords_changed=int((f10 != s["sigma"][l]).sum()))
        e = c.stats("frac8")
        frac8 = dict(tokens=e["tokens"], s_R=e["spread"], s_R_full=sp, layers={
            l: dict(mu_rel_change=round(float((e["mean"][l] - s["mean"][l]).norm() / (s["mean"][l].norm() + 1e-12)), 5),
                    ratio_cos=round(float(torch.nn.functional.cosine_similarity(
                        e["sigma"][R] / e["sigma"][l], s["sigma"][R] / s["sigma"][l], dim=0)), 5),
                    ratio_rel_change_median=round(float(((e["sigma"][R] / e["sigma"][l]) / (s["sigma"][R] / s["sigma"][l])
                                                         - 1).abs().median()), 5))
            for l in LAYERS[m]})
        per_layer = {}
        for l in LAYERS[m]:
            H = torch.load(paths.DATA / "states" / m / f"heldout_L{l}.pt")
            idx = torch.linspace(0, len(H) - 1, n_tokens).long()
            rows = {a: dict(identical=0, cos_dev=[], cos_full=[], dist=[], size=[]) for a in ABLATIONS}
            switch_off, scalar_vs_moved, shared = True, 0.0, []
            for i in idx:
                h = H[i].float()
                v = c.carry(h, l, "diag")
                switch_off &= torch.equal(v, diag_carry(h, s["mean"][l], mu_R, s["sigma"][l], s["sigma"][R], sp))
                scalar_vs_moved = max(scalar_vs_moved, float((c.carry(h, l, "diag-scalar") - c.carry(h, l, "moved")).norm()) / sp)
                # how much of D h (the uncentred direction) is the direction every token shares, D mu_l
                ratio = s["sigma"][R] / s["sigma"][l]
                shared.append(float(torch.nn.functional.cosine_similarity(h * ratio, s["mean"][l] * ratio, dim=0)))
                for a in ABLATIONS:
                    w = c.carry(h, l, f"diag-{a}")
                    r = rows[a]
                    r["identical"] += torch.equal(w, v)
                    r["cos_dev"].append(float(torch.nn.functional.cosine_similarity(w - mu_R, v - mu_R, dim=0)))
                    r["cos_full"].append(float(torch.nn.functional.cosine_similarity(w, v, dim=0)))
                    r["dist"].append(float((w - v).norm()) / sp)
                    r["size"].append(float((w - mu_R).norm()) / sp)
            med = lambda xs: round(statistics.median(xs), 5)  # noqa: E731
            per_layer[l] = dict(
                n=len(idx), nothing_ablated_is_diag_bit_for_bit=bool(switch_off),
                scalar_minus_moved_max_over_sR=scalar_vs_moved, cos_Dh_with_Dmu_l_median=med(shared),
                ablations={a: dict(identical_to_vlens=f"{r['identical']}/{len(idx)}",
                                   cos_deviation_from_mu_R_median=med(r["cos_dev"]), cos_vector_median=med(r["cos_full"]),
                                   distance_over_sR_median=med(r["dist"]), max_distance_over_sR=round(max(r["dist"]), 6),
                                   size_around_mu_R_over_sR_median=med(r["size"]))
                           for a, r in rows.items()})
        out[m] = dict(R=R, s_R=sp, mu_R_norm=float(mu_R.norm()), standard_stats_rebuilt_exactly=rebuilt, floors=floors,
                      frac8=frac8, layers=per_layer, alt_stats={a: kw for a, kw in ALT_STATS.items()})
        print(f"== {m} (R {R}, s_R {sp:.1f}, |mu_R| {float(mu_R.norm()):.1f}); standard stats rebuilt exactly: {rebuilt}")
        print("   floors bind at:", {l: (f["below_1pct"], f["below_10pct"]) for l, f in floors.items()})
        print(f"   frac8: tokens {e['tokens']}, s_R {e['spread']:.2f} (full {sp:.2f})")
        for l, row in per_layer.items():
            print(f"   L{l}: diag bit for bit {row['nothing_ablated_is_diag_bit_for_bit']}, scalar-moved "
                  f"{row['scalar_minus_moved_max_over_sR']:.1e}, cos(Dh, Dmu_l) {row['cos_Dh_with_Dmu_l_median']:.3f}")
            for a, r in row["ablations"].items():
                print(f"      {a:9s} identical {r['identical_to_vlens']:>8s}  cos dev {r['cos_deviation_from_mu_R_median']:+.4f}"
                      f"  cos {r['cos_vector_median']:+.4f}  dist/s_R {r['distance_over_sR_median']:.4f}"
                      f" (max {r['max_distance_over_sR']:.2e})  size/s_R {r['size_around_mu_R_over_sR_median']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    results.save_numbers(OUT / "check.json", out)
    print("->", OUT / "check.json")


# --- table: scripts/score_variants.py's code with the ablation rows ---------------------------------------------------
ROWS = [("Word in the request", "word", None), ("Unchanged copy", "copy", None),
        ("Moved copy (D = I; = scalar D)", "moved", None), ("V-lens", "copy:diag", None),
        ("No centering", "copy:diag-nocenter", "new"), ("No insertion-layer mean", "copy:diag-nomuR", "new"),
        ("No size normalization", "copy:diag-nosize", "new"), ("Source spreads only", "copy:diag-srconly", "new"),
        ("Insertion spreads only", "copy:diag-dstonly", "new"), ("Statistics from 1/8", "copy:diag-frac8", "new")]
REF = "V-lens"
SETS = {"val": ({"bagel": "heldout_cells", "show_o2": "heldout_cells", "sensenova": "heldout_cells_thing"},
                "lens_ablation_val_cells", "heldout_items"),
        "test": ({m: "fresh2_cells" for m in MAIN}, "lens_ablation_test_cells", "fresh2_items")}
SEED, N_BOOT = 20261012, 2000


def _scores(folder, model):
    path = R_ / folder
    if not (path / "scores.json").exists():
        return None
    return json.loads((path / "scores.json").read_text())["scores"].get(model)


def units(old_folder, new_folder, model):
    old, new = _scores(old_folder, model), _scores(new_folder, model)
    out = {}
    for name, way, src in ROWS:
        s = new if src == "new" else old
        if s is None:
            continue
        th, dm = s["two_hop"].get(way, {}).get("all"), s["double_meaning"].get(way, {}).get("all")
        if not (th and dm):
            continue
        occ = {tuple(u) for u in dm["hit_units"]}
        both = {(i, o) for i, o, _ in occ if all((i, o, q) in occ for q in ("first", "second"))}
        out[name] = dict(q_hits={u[0] for u in th["hit_units"]}, n_q=th["n"], both=both, n_s=dm["sentences"],
                         control=dict(two_hop=th["control"], double_meaning=dm["control"]),
                         literal=dict(two_hop=th["literal"], double_meaning=dm["literal"]))
    return out


def table_one(key, rng):
    old_sets, new_folder, items = SETS[key]
    print(f"\n== {key}")
    data = {m: units(old_sets[m], new_folder, m) for m in MAIN}
    data = {m: d for m, d in data.items() if d}
    names = [n for n, _, _ in ROWS if any(n in d for d in data.values())]
    out = dict(counts={}, scores={})
    print(" " * 32 + "".join(f"{m:>32}" for m in data))
    print(" " * 32 + "".join(f"{'two-step':>16}{'both meanings':>16}" for m in data))
    for n in names:
        line = n.ljust(32)
        for m, d in data.items():
            r = d.get(n)
            if r is None:
                line += f"{'-':>16}{'-':>16}"
                continue
            line += f"{len(r['q_hits']):>11}/{r['n_q']:<4}{len(r['both']):>11}/{r['n_s']:<4}"
            out["counts"].setdefault(n, {})[m] = dict(two_hop=[len(r["q_hits"]), r["n_q"]], both=[len(r["both"]), r["n_s"]],
                                                     control=r["control"], literal=r["literal"])
        print(line)
    for m, d in data.items():
        sizes = {(r["n_q"], r["n_s"]) for r in d.values()}
        if len(sizes) != 1:
            print(f"   WARNING {m}: rows count different units {sizes} (drawing or judging not complete?)")
    q_index = {}
    for m in data:
        recs = []
        for f in (R_ / old_sets[m] / m).glob("records*.json"):
            recs += [r for r in json.loads(f.read_text()) if r["kind"] == "two_hop" and r["items"] == items]
        q_index[m] = sorted({r["item"] for r in recs})

    def score(n, sample=None):
        rates = []
        for m in MAIN:
            r = data.get(m, {}).get(n)
            if r is None:
                return None
            if sample is None:
                rates += [len(r["q_hits"]) / r["n_q"], len(r["both"]) / r["n_s"]]
            else:
                qs, ws = sample[m]
                rates += [sum(q in r["q_hits"] for q in qs) / len(qs),
                          sum(((w, "ab") in r["both"]) + ((w, "ba") in r["both"]) for w in ws) / (2 * len(ws))]
        return statistics.mean(rates)

    full = {n: score(n) for n in names}
    boots = {n: [] for n in names if full[n] is not None}
    if all(m in data for m in MAIN):
        pops = {m: next(iter(data[m].values()))["n_s"] // 2 for m in MAIN}
        for _ in range(N_BOOT):
            sample = {m: ([rng.choice(q_index[m]) for _ in q_index[m]],
                          [rng.choice(range(pops[m])) for _ in range(pops[m])]) for m in MAIN}
            for n in boots:
                boots[n].append(score(n, sample))
    print(f"   score, three models (six rates); 95% paired bootstrap interval of the difference from the {REF}:")
    for n in names:
        if full[n] is None:
            continue
        entry = dict(score=round(full[n], 4))
        if boots.get(REF) and n != REF:
            d = sorted(a - b for a, b in zip(boots[n], boots[REF]))
            entry["minus_vlens"] = round(full[n] - full[REF], 4)
            entry["minus_vlens_95"] = [round(d[int(0.025 * len(d))], 4), round(d[int(0.975 * len(d)) - 1], 4)]
        out["scores"][n] = entry
        print(f"   {n:32s} {full[n]:.3f}" + (f"   minus V-lens {entry['minus_vlens']:+.3f} "
                                             f"[{entry['minus_vlens_95'][0]:+.3f}, {entry['minus_vlens_95'][1]:+.3f}]"
                                             if "minus_vlens" in entry else ""))
    return out


def table():
    rng = random.Random(SEED)  # one generator, Val then Test, as scripts/score_variants.py: the stored rows reproduce its intervals
    numbers = {key: table_one(key, rng) for key in ("val", "test")}
    numbers["not_drawn"] = {
        "scalar D": "identical to the moved copy (one factor cancels in D x / |D x|; check.json scalar_minus_moved)",
        "variance floor 0": "identical to the V-lens bit for bit (the 1% floor binds on no coordinate; check.json floors)",
        "variance floor 10%": "identical to the V-lens except one coordinate at one SenseNova-U1 layer (check.json)"}
    OUT.mkdir(parents=True, exist_ok=True)
    results.save_numbers(OUT / "table.json", numbers)
    print("->", OUT / "table.json")


def main():
    ap = argparse.ArgumentParser(description="Lens ablation: statistics, CPU checks and the table.")
    ap.add_argument("step", choices=["stats", "check", "table"])
    args = ap.parse_args()
    {"stats": stats, "check": check, "table": table}[args.step]()


if __name__ == "__main__":
    main()
