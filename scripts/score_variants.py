"""The lens-variant tables: Table 1, Supp. Table S2 and the final-norm comparison of Supp. Table S3 (CPU, no model).

Every variant carries the same states to the same insertion layer and is drawn with the same seeds, requests and judges
(scripts/draw_cells.py ways: word, copy, moved, copy:rel, copy:pct, copy:diag = the V-lens, vlens:<transport> = the
Jacobian map, vlens:<transport>_selfgain = the Jacobian map with gain, fnorm and fnorm:sR = the final normalization of
the drawing pathway, applied as Diffusion Lens applies the text encoder's final norm, without or with the V-lens radius).
Two steps:
- scale: the gains of each model's final norms and the size and direction of the normed state against the states the
  insertion layer normally holds, from the saved text states (scripts/collect_states.py) and the V-lens statistics
  (Supp. A.4). -> fnorm/scale.json
- table: the Val and Test rows per model (two-step questions whose object is drawn, sentences with both meanings), the
  six-rate score of the three models (Table S2), the mean of the three two-step rates (Table 1), and the paired bootstrap
  interval of every row minus the V-lens (2,000 resamples of questions and words, seed 20261012). The rows are read
  from the scores scripts/score_cells.py writes: Val = heldout_cells (BAGEL, Show-o2) and heldout_cells_thing
  (SenseNova-U1, request "thing"), Test = fresh2_cells; the final-norm rows from fnorm_val_cells and fnorm_test_cells.
  -> fnorm/table.json

    python scripts/score_variants.py scale
    python scripts/score_variants.py table
"""

import argparse
import json
import random
import statistics

import torch

from vlens import paths, results

R_ = results.RESULTS
OUT = R_ / "fnorm"
MAIN = ["bagel", "show_o2", "sensenova"]
LAYERS = {"bagel": [2, 4, 6, 8, 10, 14, 20], "show_o2": [6, 10, 14, 20, 24, 27], "sensenova": [8, 12, 16, 20, 24, 28, 36],
          "hunyuan": [0, 6, 9, 12, 27, 30]}  # HunyuanImage: scale check only (not in the paper)


def norm_weights(model: str) -> dict:
    """The final RMSNorm gains of a model, read from its weight files (no model): the drawing pathway's and the text's."""
    from safetensors import safe_open
    if model == "bagel":
        with safe_open(str(paths.snapshot("ByteDance-Seed/BAGEL-7B-MoT") / "ema.safetensors"), "pt") as f:
            return dict(draw=f.get_tensor("language_model.model.norm_moe_gen.weight"),
                        text=f.get_tensor("language_model.model.norm.weight"), eps=1e-6)
    if model == "sensenova":
        snap = paths.snapshot("sensenova/SenseNova-U1-8B-MoT")
        idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
        out = {}
        for key, name in (("draw", "language_model.model.norm_mot_gen.weight"), ("text", "language_model.model.norm.weight")):
            with safe_open(str(snap / idx[name]), "pt") as f:
                out[key] = f.get_tensor(name)
        return out | dict(eps=1e-6)
    if model == "show_o2":
        snap = paths.snapshot("showlab/show-o2-7B")
        idx = json.loads((snap / "diffusion_pytorch_model.bin.index.json").read_text())["weight_map"]
        sd = torch.load(snap / idx["showo.model.norm.weight"], map_location="cpu", mmap=True, weights_only=True)
        w = sd["showo.model.norm.weight"].clone()
        return dict(draw=w, text=w, eps=1e-6)
    if model == "hunyuan":  # no per-token final norm on the drawing path; ln_f (the text head's) is the closest one
        snap = paths.MODELS / "HunyuanImage3Instruct"
        idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(str(snap / idx["model.ln_f.weight"]), "pt") as f:
            w = f.get_tensor("model.ln_f.weight")
        return dict(draw=w, text=w, eps=1e-5)
    raise ValueError(model)


def rms_norm(h: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    h = h.float()
    return w.float() * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))


def scale(models):
    out = {}
    for m in models:
        w = norm_weights(m)
        st = torch.load(paths.LENS_STORE / f"{m}_copy_stats" / "stats.pt", map_location="cpu", weights_only=False)
        R = st["R"]
        folder = paths.DATA / "states" / m
        hR = torch.load(folder / f"calib_L{R}.pt").float()
        mu_R = st["mean"][R].float()
        row = dict(R=R, s_R=float(st["spread"]), mu_R_norm=float(mu_R.norm()),
                   state_at_R_norm_median=float(hR.norm(dim=1).median()),
                   gain=dict(draw=dict(mean=float(w["draw"].float().mean()), min=float(w["draw"].float().min()),
                                       max=float(w["draw"].float().max()), rms=float(w["draw"].float().pow(2).mean().sqrt())),
                             text=dict(mean=float(w["text"].float().mean()), min=float(w["text"].float().min()),
                                       max=float(w["text"].float().max()), rms=float(w["text"].float().pow(2).mean().sqrt()))),
                   draw_equals_text=bool(torch.equal(w["draw"], w["text"])), layers={})
        for l in LAYERS[m]:
            f = folder / f"calib_L{l}.pt"
            if not f.exists():
                row["layers"][l] = None
                continue
            h = torch.load(f).float()
            n = rms_norm(h, w["draw"], w["eps"])
            c = lambda a, b: torch.nn.functional.cosine_similarity(a, b[None].expand_as(a), dim=1)  # noqa: E731
            row["layers"][l] = dict(
                state_norm_median=float(h.norm(dim=1).median()),
                fnorm_norm_median=float(n.norm(dim=1).median()),
                fnorm_over_state_at_R=float(n.norm(dim=1).median() / hR.norm(dim=1).median()),
                # how much of the insertion layer's average direction the normed state carries, and the state's own
                cos_fnorm_mu_R_median=float(c(n, mu_R).median()),
                cos_state_at_R_mu_R_median=float(c(hR, mu_R).median()))
        out[m] = row
        print(m, json.dumps({k: v for k, v in row.items() if k != "layers"}))
        for l, v in row["layers"].items():
            print("   L", l, v)
    OUT.mkdir(parents=True, exist_ok=True)
    results.save_numbers(OUT / "scale.json", out)
    print("->", OUT / "scale.json")


# --- table ---------------------------------------------------------------------------------------------------------
STD = {"bagel": "bagel_transport_to8_n64", "show_o2": "show_o2_transport_to27_n64",
       "sensenova": "sensenova_transport_to24_n64"}
ROWS = [("Word in the request", "word", None), ("Unchanged copy", "copy", None), ("Moved copy", "moved", None),
        ("Relative size", "copy:rel", None), ("Percentile size", "copy:pct", None), ("V-lens", "copy:diag", None),
        ("Jacobian map", "vlens:{std}", None), ("Jacobian map with gain", "vlens:{std}_selfgain", None),
        ("Final norm", "fnorm", "new"), ("Final norm at s_R", "fnorm:sR", "new")]
SETS = {"val": ({"bagel": "heldout_cells", "show_o2": "heldout_cells", "sensenova": "heldout_cells_thing"},
                "fnorm_val_cells", "heldout_items"),
        "test": ({m: "fresh2_cells" for m in MAIN}, "fnorm_test_cells", "fresh2_items")}
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
        way = way.format(std=STD[model])
        th, dm = s["two_hop"].get(way, {}).get("all"), s["double_meaning"].get(way, {}).get("all")
        if not (th and dm):
            continue
        occ = {tuple(u) for u in dm["hit_units"]}
        both = {(i, o) for i, o, _ in occ if all((i, o, q) in occ for q in ("first", "second"))}
        out[name] = dict(q_hits={u[0] for u in th["hit_units"]}, n_q=th["n"], both=both, n_s=dm["sentences"],
                         control=th["control"], literal=th["literal"])
    return out


def table_one(key, rng):
    old_sets, new_folder, items = SETS[key]
    print(f"\n== {key}")
    data = {m: units(old_sets[m], new_folder, m) for m in MAIN}
    data = {m: d for m, d in data.items() if d}
    names = [n for n, _, _ in ROWS if any(n in d for d in data.values())]
    out = dict(counts={}, scores={})
    for n in names:
        line = n.ljust(24)
        for m, d in data.items():
            r = d.get(n)
            if r is None:
                line += f"{'-':>16}{'-':>16}"
                continue
            line += f"{len(r['q_hits']):>11}/{r['n_q']:<4}{len(r['both']):>11}/{r['n_s']:<4}"
            out["counts"].setdefault(n, {})[m] = dict(two_hop=[len(r["q_hits"]), r["n_q"]], both=[len(r["both"]), r["n_s"]])
        print(line)
    # every row of a model must count the same questions and sentences (the final-norm pictures cover the whole set)
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
    print("   score, three models (six rates); 95% interval of the difference from the V-lens:")
    for n in names:
        if full[n] is None:
            continue
        entry = dict(score=round(full[n], 4))
        if all(n in data.get(m, {}) for m in MAIN):  # Table 1: the mean of the three two-step rates
            entry["two_hop_mean"] = round(statistics.mean(len(data[m][n]["q_hits"]) / data[m][n]["n_q"] for m in MAIN), 4)
        if boots.get("V-lens") and n != "V-lens":
            d = sorted(a - b for a, b in zip(boots[n], boots["V-lens"]))
            entry["minus_vlens"] = round(full[n] - full["V-lens"], 4)
            entry["minus_vlens_95"] = [round(d[int(0.025 * len(d))], 4), round(d[int(0.975 * len(d)) - 1], 4)]
        out["scores"][n] = entry
        print(f"   {n:24s} {full[n]:.3f}" + (f"   two-step mean {entry['two_hop_mean']:.3f}" if "two_hop_mean" in entry
                                             else "") + (f"   minus V-lens {entry['minus_vlens']:+.3f} "
                                             f"[{entry['minus_vlens_95'][0]:+.3f}, {entry['minus_vlens_95'][1]:+.3f}]"
                                             if "minus_vlens" in entry else ""))
    return out


def table():
    rng = random.Random(SEED)
    numbers = {key: table_one(key, rng) for key in ("val", "test")}
    OUT.mkdir(parents=True, exist_ok=True)
    results.save_numbers(OUT / "table.json", numbers)
    print("->", OUT / "table.json")


def main():
    ap = argparse.ArgumentParser(description="Lens-variant tables (Table 1, S2, S3) and the final-norm size check.")
    ap.add_argument("step", choices=["scale", "table"])
    ap.add_argument("--models", default=",".join(MAIN),
                    help="scale: models to check (hunyuan also works once its statistics and states exist)")
    args = ap.parse_args()
    if args.step == "scale":
        scale(args.models.split(","))
    else:
        table()


if __name__ == "__main__":
    main()
