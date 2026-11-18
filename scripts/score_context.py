"""Score the context tests (paper Table 2, Supp. Tables S7 and S8, Sec. 4.5 and Supp. I). CPU only.

Reads the pictures' records (scripts/draw_cells.py, folder followup_cells) and the panel's main-subject answers
(scripts/run_judge.py --set cells:followup_cells --subject; answer set cells_followup_cells_subject).
- a picture succeeds when at least two of the three judges choose its own thing alone;
- a state (text, word position, layer, way) succeeds when at least 2 of its 3 seeds succeed;
- A (descriptions of unnamed objects), B (pronoun reference): a pair succeeds at a layer when both of its texts succeed
  there; C (words with two meanings): a sentence succeeds when both occurrences succeed at the same layer, and a word's
  score is the mean over its two sentences (both orders);
- primary: pair (A, B) or word (C) success at the insertion layer, per family and model, way copy:diag (the V-lens);
  chance by 10,000 swaps of the two labels within pairs (words for C), one-sided p = (1 + n_>=) / (1 + 10,000). The
  paper reports these raw p-values; a Holm-adjusted p over the tests of one run is also stored (p_holm);
- secondary: the three-layer band average, each layer, any layer; B by subtype (G01-G10 verb changed, G11-G20 verb
  held); C first and second occurrence; the word on its own; the other ways (copy, vlens) at the same layers; 95%
  intervals from 2,000 bootstrap samples of pairs (words); the selecting-clause-alone controls of B (Table S8).
Writes $VLENS_OUT/followup_cells/scores.json and prints the summary.

    python scripts/score_context.py                                              # BAGEL, Show-o2, SenseNova-U1
    python scripts/score_context.py --models hunyuan --out scores_hunyuan.json  # HunyuanImage-3.0
    python scripts/score_context.py --models liquid,janus_pro --out scores_autoregressive.json   # Liquid, Janus-Pro
"""

import json
from collections import defaultdict

import numpy as np
import yaml

from vlens import prompts, results
from vlens.judges import answer_files

R = results.RESULTS
FOLDER = R / "followup_cells"
JUDGES = R / "judges"
PANEL = ["qwen3vl", "gemma3", "llavaov2"]
BAND = {"bagel": [6, 8, 14], "show_o2": [20, 24, 27], "sensenova": [20, 24, 32]}
INSERT = {"bagel": 8, "show_o2": 27, "sensenova": 24}
# HunyuanImage-3.0 and the autoregressive models, scored by the same rules in their own files (--models, --out)
EXTRA_BAND = {"liquid": [6, 10, 14], "janus_pro": [8, 12, 16], "hunyuan": [9, 12, 15]}
EXTRA_INSERT = {"liquid": 6, "janus_pro": 8, "hunyuan": 9}
MAIN = "copy:diag"
N_BOOT, N_SWAP, ALPHA = 2000, 10000, 0.05


def load_answers(set_name):
    out = defaultdict(dict)
    for judge in PANEL:
        for f in answer_files(JUDGES, judge, set_name):
            for a in json.loads(f.read_text()):
                out[a["file"]][judge] = a
    return out


def holm(pvals: dict) -> dict:
    """Holm-adjusted p-values (step-down, monotone)."""
    keys = sorted(pvals, key=pvals.get)
    m, adj, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default="", help="score these models (comma separated) instead of BAGEL, Show-o2 and "
                                                  "SenseNova-U1")
    ap.add_argument("--out", default="scores.json", help="file name in the cells folder")
    # another drawing of the same items: its cells folder ($VLENS_OUT/<folder>/, answers cells:<folder>_subject), and
    # optionally another insertion layer and band (not used for the paper's numbers)
    ap.add_argument("--folder", default="followup_cells")
    ap.add_argument("--insert", type=int, default=None, help="with one model: score at this insertion layer")
    ap.add_argument("--band", default="", help="with one model: these layers (comma separated)")
    args = ap.parse_args()
    global FOLDER
    FOLDER = R / args.folder
    if args.models:
        names = args.models.split(",")
        main_band = dict(BAND)
        BAND.clear()
        BAND.update({m: EXTRA_BAND.get(m) or main_band[m] for m in names})
        INSERT.update({m: EXTRA_INSERT[m] for m in names if m in EXTRA_INSERT})
    if args.insert is not None or args.band:
        assert len(BAND) == 1, "--insert and --band need exactly one model (--models)"
        m = next(iter(BAND))
        if args.band:
            BAND[m] = [int(x) for x in args.band.split(",")]
        if args.insert is not None:
            INSERT[m] = args.insert
    meta = {f: yaml.safe_load((prompts.PROMPTS / f"{f}.yaml").read_text())
            for f in ("followup_matched", "followup_reference", "followup_meanings")}
    ans = load_answers(f"cells:{args.folder}_subject")
    recs = []
    for m in BAND:
        seen = set()
        for f in sorted((FOLDER / m).glob("records*.json")):
            for r in json.loads(f.read_text()):
                if r["file"] not in seen:
                    seen.add(r["file"])
                    recs.append(r)
    judged = [r for r in recs if len(ans.get(r["file"], {})) == len(PANEL)]
    print(f"{len(judged)} of {len(recs)} pictures judged by all {len(PANEL)} judges")

    pic = {}
    for r in judged:
        a = ans[r["file"]]
        truth, alt = next(iter(a.values()))["truth"], next(iter(a.values()))["literal"]
        pic[r["file"]] = dict(own=sum(x["chosen"] == truth for x in a.values()) >= 2,
                              other=sum(x["chosen"] == alt for x in a.values()) >= 2)
    seeds = defaultdict(lambda: defaultdict(list))
    for r in judged:
        key = (r["model"], r["items"], r["item"], r.get("order"), r["position"], r["layer"], r["way"])
        seeds[key]["own"].append(pic[r["file"]]["own"])
        seeds[key]["other"].append(pic[r["file"]]["other"])
    st = {k: dict(own=sum(v["own"]) >= 2, other=sum(v["other"]) >= 2, n=len(v["own"])) for k, v in seeds.items()}

    def units(model, way, layer):
        """{(family, unit id): (own, other, extra)} at one layer; own/other are 0/1 for pairs, the mean over the two
        sentences for words. extra: B subtype, or C (first own, second own) counts."""
        out = {}
        for fam, items in (("A", "followup_matched"), ("B", "followup_reference")):
            rows = meta[items]["reads"]
            pairs = defaultdict(dict)
            for i, it in enumerate(rows):
                if it["role"] in ("text1", "text2"):
                    pairs[it["pair"]][it["role"]] = i
            for pid, roles in pairs.items():
                pos = rows[roles["text1"]]["reads"][0]["name"]
                k1 = (model, items, roles["text1"], None, pos, layer, way)
                k2 = (model, items, roles["text2"], None, pos, layer, way)
                if k1 in st and k2 in st:
                    sub = ("verb-changed" if int(pid[1:]) <= 10 else "verb-held") if fam == "B" else None
                    out[(fam, pid)] = (float(st[k1]["own"] and st[k2]["own"]), float(st[k1]["other"] and st[k2]["other"]),
                                       sub)
        for i, it in enumerate(meta["followup_meanings"]["double_meaning"]):
            cells = {(o, p): st.get((model, "followup_meanings", i, o, p, layer, way))
                     for o in ("ab", "ba") for p in ("first", "second")}
            if all(cells.values()):
                own = np.mean([cells[(o, "first")]["own"] and cells[(o, "second")]["own"] for o in ("ab", "ba")])
                oth = np.mean([cells[(o, "first")]["other"] and cells[(o, "second")]["other"] for o in ("ab", "ba")])
                first = sum(cells[(o, "first")]["own"] for o in ("ab", "ba"))
                second = sum(cells[(o, "second")]["own"] for o in ("ab", "ba"))
                out[("C", it["word"])] = (float(own), float(oth), (first, second))
        return out

    rng = np.random.default_rng(20261019)
    primary, secondary = {}, {}
    for model in BAND:
        for way in sorted({k[6] for k in st if k[0] == model and k[6] != "word"}):
            by_layer = {l: units(model, way, l) for l in BAND[model]}
            for fam in ("A", "B", "C"):
                ins = {k: v for k, v in by_layer[INSERT[model]].items() if k[0] == fam}
                if not ins:
                    continue
                keys = sorted(ins)
                own = np.array([ins[k][0] for k in keys])
                oth = np.array([ins[k][1] for k in keys])
                stat = float(own.mean())
                swaps = rng.random((N_SWAP, len(keys))) < 0.5
                null = np.where(swaps, oth, own).mean(1)
                p = float((1 + np.sum(null >= stat - 1e-9)) / (1 + N_SWAP))
                boot = [own[rng.integers(0, len(keys), len(keys))].mean() for _ in range(N_BOOT)]
                band = {l: {k: v for k, v in by_layer[l].items() if k[0] == fam} for l in BAND[model]}
                band_mean = {l: float(np.mean([v[0] for v in band[l].values()])) if band[l] else None for l in band}
                any_layer = float(np.mean([max(band[l].get(k, (0,))[0] for l in BAND[model]) for k in keys]))
                row = dict(n_units=len(keys), success_insertion=stat, count_insertion=float(own.sum()),
                           ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                           chance=float(null.mean()), p=p, by_layer=band_mean,
                           band_average=float(np.mean([v for v in band_mean.values() if v is not None])),
                           any_layer=any_layer, units={k[1]: ins[k][0] for k in keys})
                if fam == "B":
                    row["by_subtype"] = {s: f"{int(sum(ins[k][0] for k in keys if ins[k][2] == s))}/"
                                            f"{sum(ins[k][2] == s for k in keys)}" for s in ("verb-changed", "verb-held")}
                if fam == "C":
                    row["first_second_own"] = [int(sum(ins[k][2][0] for k in keys)), int(sum(ins[k][2][1] for k in keys)),
                                               2 * len(keys)]
                (primary if way == MAIN else secondary)[f"{model}|{way}|{fam}"] = row
    adj = holm({k: v["p"] for k, v in primary.items()})
    for k in primary:
        primary[k]["p_holm"] = adj[k]
        primary[k]["confirmed"] = adj[k] < ALPHA

    # the word on its own (drawn once per text, at the first band layer): does it already show the target?
    word = defaultdict(lambda: [0, 0])
    for k, v in st.items():
        if k[6] == "word":
            word[(k[0], k[1])][0] += v["own"]
            word[(k[0], k[1])][1] += 1
    # verb-only controls: which thing the panel's majority picks (copy:diag, insertion layer)
    controls = defaultdict(lambda: defaultdict(int))
    rows = meta["followup_reference"]["reads"]
    for r in judged:
        if r["items"] == "followup_reference" and r["way"] == MAIN and r["layer"] == INSERT[r["model"]] \
                and rows[r["item"]]["role"] == "verb_only":
            chosen = [x["chosen"] for x in ans[r["file"]].values()]
            top = max(set(chosen), key=chosen.count)
            controls[r["model"]]["named thing" if top == rows[r["item"]]["truth"] else
                                 "other thing" if top == rows[r["item"]]["alt"] else top] += 1
    scores = dict(primary=primary, secondary=secondary,
                  word_alone={f"{m}|{i}": f"{a}/{n}" for (m, i), (a, n) in word.items()},
                  verb_only_controls={m: dict(v) for m, v in controls.items()},
                  n_judged=len(judged), n_pictures=len(recs))
    (FOLDER / args.out).write_text(json.dumps(scores, indent=1))
    for k, v in sorted(primary.items()):
        extra = v.get("by_subtype") or v.get("first_second_own") or ""
        print(f"{k:28s} n={v['n_units']:2d} insertion {v['success_insertion']:.2f} ({v['count_insertion']:g}) "
              f"[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] chance {v['chance']:.2f} p={v['p']:.4f} Holm {v['p_holm']:.4f} "
              f"{'CONFIRMED' if v['confirmed'] else ''}  band {v['band_average']:.2f} any {v['any_layer']:.2f} {extra}")
    for k, v in sorted(secondary.items()):
        print(f"  {k:28s} insertion {v['success_insertion']:.2f} band {v['band_average']:.2f}")
    print("word alone:", scores["word_alone"])
    print("verb-only controls:", scores["verb_only_controls"])


if __name__ == "__main__":
    main()
