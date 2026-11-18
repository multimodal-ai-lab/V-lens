"""Score lens tuning on pronouns (Table 3, Supp. Tables S14 to S18 for BAGEL and SenseNova-U1, and the HunyuanImage
numbers of Supp. F.1). CPU only.

- Lens pictures ($VLENS_OUT/vsup_cells/<model>_<arm>[_s<seed>]/, judged with the main-subject question,
  scripts/run_judge.py --set cells:vsup_cells --subject): a picture succeeds when at least two judges choose the referent
  alone; a text when 2 of its 3 drawing seeds succeed; a unit (one noun pair in one noun order: the two texts with the
  two referents) when both texts succeed. Arms with three training seeds are averaged over seeds per unit. Paired
  bootstrap (10,000) over noun pairs (both noun orders resampled together) for each arm against arm 0, and V against R
  (word state), T and G.
- Text answers (vsup_pilot/text_<arm>_s<seed>.json from scripts/lens_tuning.py text): the trained question ("name") and
  the untrained ones (choice, yes, no); the untrained score of a text is the mean of the three. Paired bootstrap.
- Preservation: the general questions, and the texts that arm 0 already draws correctly ("retained").
- Ordinary drawing ($VLENS_OUT/vsup_gen/, judged the same way, --set cells:vsup_gen --subject): a text succeeds when both
  drawing seeds show the referent alone; pooled over training seeds, and per pair of texts.
Table 3 and Supp. Table S14 read lens[arm][test1 | test2 | test3] (Phrases | Verbs | Nouns), "tests" and "dev";
Supp. Table S15 text[arm] (name, untrained, general) and preservation; Supp. Table S16 generation_pooled; Supp. Table S17
lens[arm]["per_seed_tests"] and the per-folder generation rows; Supp. Table S18 the vs_0 / V_vs_* intervals.
The random stream of the intervals is consumed in the order of the arms found, so the intervals reproduce the paper's
with exactly the arms 0, G, R, S, T, V (HunyuanImage: 0, G, T, V) in the output folders.
Writes $VLENS_OUT/vsup_pilot/scores.json (scores_sensenova.json, scores_hunyuan.json) and prints a summary.

    python scripts/score_lens_tuning.py
    python scripts/score_lens_tuning.py --model sensenova
    python scripts/score_lens_tuning.py --model hunyuan
"""

import argparse

import json
from collections import Counter, defaultdict

import numpy as np
import yaml

from vlens import prompts, results

R = results.RESULTS
PILOT = R / "vsup_pilot"
JUDGES = R / "judges"
PANEL = ["qwen3vl", "gemma3", "llavaov2"]
TESTS = ("test1", "test2", "test3")
GROUPS = ("dev",) + TESTS
N_BOOT = 10000
rng = np.random.default_rng(20261019)


def answers(set_tag):
    """{file: {judge: answer}} for one picture set (all folders)."""
    out = defaultdict(dict)
    for j in PANEL:
        for f in sorted(JUDGES.glob(f"answers_{j}_{set_tag}_*.json")):
            for a in json.loads(f.read_text()):
                out[a["file"]][j] = a
    return out


def records(folder, model):
    out = {}
    for d in sorted(p for p in folder.iterdir() if p.is_dir() and p.name.startswith(model + "_")):
        seen = {}
        for f in sorted(d.glob("records_*.json")):
            for r in json.loads(f.read_text()):
                seen[r["file"]] = r
        out[d.name] = list(seen.values())
    return out


def arm_of(folder):
    parts = folder.split("_")  # bagel_0, bagel_T_s0
    return parts[1], (int(parts[2][1:]) if len(parts) > 2 else None)


def boot_diff(a, b, clusters=None):
    """Mean of a - b with a 95% bootstrap interval; with `clusters` (one label per element), whole clusters are
    resampled (a noun pair appears as two units, one per noun order, and they move together)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if clusters is None:
        idx = rng.integers(0, len(d), (N_BOOT, len(d)))
        m = d[idx].mean(1)
    else:
        labels = sorted(set(clusters))
        groups = [np.flatnonzero(np.asarray(clusters) == c) for c in labels]
        picks = rng.integers(0, len(groups), (N_BOOT, len(groups)))
        m = np.array([np.concatenate([d[groups[g]] for g in row]).mean() for row in picks])
    return dict(diff=float(d.mean()), ci95=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))])


def boot_diff_with(gen, a, b, clusters):
    """boot_diff with its own random stream (the pair-level intervals must not shift the others)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    labels = sorted(set(clusters))
    groups = [np.flatnonzero(np.asarray(clusters) == c) for c in labels]
    picks = gen.integers(0, len(groups), (N_BOOT, len(groups)))
    m = np.array([np.concatenate([d[groups[g]] for g in row]).mean() for row in picks])
    return dict(diff=float(d.mean()), ci95=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))])


def pair_of(unit):
    return unit.rsplit("_o", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bagel", choices=["bagel", "sensenova", "hunyuan"])
    a = ap.parse_args()
    model = a.model
    cells_set, gen_set, cells_dir, gen_dir = "vsup_cells", "vsup_gen", R / "vsup_cells", R / "vsup_gen"
    global GROUPS
    if model == "hunyuan":   # HunyuanImage drew the test groups only (no dev group)
        GROUPS = TESTS
    base = f"{model}_0"
    text_prefix = "text_" if model == "bagel" else f"text_{model}_"
    data = yaml.safe_load((prompts.PROMPTS / "vsup_items.yaml").read_text())
    reads = data["reads"]
    unit_of = {i: it["unit"] for i, it in enumerate(reads)}
    group_of = {i: it["group"] for i, it in enumerate(reads)}
    units = {g: sorted({it["unit"] for it in reads if it["group"] == g}) for g in GROUPS}
    out = {}

    # ---------------------------------------------------------------- lens pictures
    ans = answers(f"cells_{cells_set}_subject")
    lens_text = {}  # folder -> {item: success}
    lens_score = {}  # folder -> {item: share of judge votes for the referent}
    choices = defaultdict(Counter)
    for folder, recs in records(cells_dir, model).items():
        per = defaultdict(list)
        votes = defaultdict(list)
        for r in recs:
            a = ans.get(r["file"], {})
            if len(a) < len(PANEL):
                continue
            chosen = [x["chosen"] for x in a.values()]
            per[r["item"]].append(sum(c == r["truth"] for c in chosen) >= 2)
            votes[r["item"]].append(sum(c == r["truth"] for c in chosen) / len(PANEL))
            top, n = Counter(chosen).most_common(1)[0]
            choices[folder]["referent" if top == r["truth"] and n >= 2 else
                            "other thing" if n >= 2 and top == a[PANEL[0]]["literal"] else
                            top.split(",")[0] if n >= 2 else "no majority"] += 1
        lens_text[folder] = {i: sum(v) >= 2 for i, v in per.items() if len(v) == 3}
        lens_score[folder] = {i: float(np.mean(v)) for i, v in votes.items() if len(v) == 3}

    def unit_success(text_ok):
        u = defaultdict(list)
        for i, ok in text_ok.items():
            u[unit_of[i]].append(ok)
        return {k: float(all(v)) for k, v in u.items() if len(v) == 2}

    arms = defaultdict(dict)  # arm -> seed -> {unit: success}
    arms_text = defaultdict(dict)
    for folder, t in lens_text.items():
        arm, seed = arm_of(folder)
        arms[arm][seed] = unit_success(t)
        arms_text[arm][seed] = t

    def arm_units(arm, us):
        """Per unit, the mean over training seeds (None if a unit is missing for some seed)."""
        seeds = arms[arm]
        return np.array([np.mean([seeds[s][u] for s in seeds]) for u in us])

    lens = {}
    test_units = [u for g in TESTS for u in units[g]]
    for arm in sorted(arms):
        row = {g: float(arm_units(arm, units[g]).mean()) for g in GROUPS}
        row["tests"] = float(arm_units(arm, test_units).mean())
        row["texts_tests"] = float(np.mean([np.mean([arms_text[arm][s][i] for s in arms_text[arm]])
                                            for i in range(len(reads)) if group_of[i] in TESTS
                                            and all(i in arms_text[arm][s] for s in arms_text[arm])]))
        row["seeds"] = sorted(s for s in arms[arm] if s is not None) or ["none"]
        if arm != "0":
            row["vs_0_tests"] = boot_diff(arm_units(arm, test_units), arm_units("0", test_units),
                                          [pair_of(u) for u in test_units])
            row["vs_0_test3"] = boot_diff(arm_units(arm, units["test3"]), arm_units("0", units["test3"]),
                                          [pair_of(u) for u in units["test3"]])
            if "dev" in units:
                row["vs_0_dev"] = boot_diff(arm_units(arm, units["dev"]), arm_units("0", units["dev"]),
                                            [pair_of(u) for u in units["dev"]])
            row["per_seed_tests"] = {s: float(np.mean([arms[arm][s][u] for u in test_units])) for s in arms[arm]}
        lens[arm] = row
    out["lens"] = lens
    out["lens_choices"] = {f: dict(c) for f, c in sorted(choices.items())}
    if "V" in arms and "R" in arms:
        out["lens_V_vs_R_tests"] = boot_diff(arm_units("V", test_units), arm_units("R", test_units),
                                             [pair_of(u) for u in test_units])
    if "V" in arms and "T" in arms:
        out["lens_V_vs_T_tests"] = boot_diff(arm_units("V", test_units), arm_units("T", test_units),
                                             [pair_of(u) for u in test_units])
    if "V" in arms and "G" in arms:
        out["lens_G_vs_V_tests"] = boot_diff(arm_units("G", test_units), arm_units("V", test_units),
                                             [pair_of(u) for u in test_units])

    # ---------------------------------------------------------------- text answers
    texts = defaultdict(dict)
    for f in sorted(PILOT.glob(f"{text_prefix}*.json")):
        tag = f.stem[len(text_prefix):]
        if not (tag == "0" or tag.split("_")[0] in "TVRSG" and "_s" in tag):
            continue  # another model's file
        arm, seed = (tag, None) if tag == "0" else (tag.split("_")[0], int(tag.split("_s")[1]))
        texts[arm][seed] = json.loads(f.read_text())
    idx = {it["id"]: i for i, it in enumerate(reads)}
    text_out = {}

    def text_units(arm, key, us):
        vals = defaultdict(list)
        for s, d in texts[arm].items():
            per = defaultdict(list)
            for r in d["rows"]:
                if r["group"] in TESTS + ("dev",):
                    v = np.mean([r["choice"], r["yes"], r["no"]]) if key == "untrained" else float(r[key])
                    per[r["unit"]].append(v)
            for u, v in per.items():
                vals[u].append(np.mean(v))
        return np.array([np.mean(vals[u]) for u in us])

    for arm in sorted(texts):
        row = {}
        for key in ("name", "choice", "yes", "no", "untrained"):
            row[key] = float(text_units(arm, key, test_units).mean())
        row["untrained_test3"] = float(text_units(arm, "untrained", units["test3"]).mean())
        row["general"] = float(np.mean([d["summary"]["general"] for d in texts[arm].values()]))
        if arm != "0":
            cl = [pair_of(u) for u in test_units]
            row["untrained_vs_0"] = boot_diff(text_units(arm, "untrained", test_units), text_units("0", "untrained", test_units), cl)
            row["name_vs_0"] = boot_diff(text_units(arm, "name", test_units), text_units("0", "name", test_units), cl)
        if arm == "G" and "V" in texts:
            cl = [pair_of(u) for u in test_units]
            row["untrained_vs_V"] = boot_diff(text_units("G", "untrained", test_units), text_units("V", "untrained", test_units), cl)
            row["name_vs_V"] = boot_diff(text_units("G", "name", test_units), text_units("V", "name", test_units), cl)
        text_out[arm] = row
    out["text"] = text_out

    # ---------------------------------------------------------------- preservation
    base_ok = [i for i, ok in lens_text.get(base, {}).items() if ok and group_of[i] in TESTS]
    out["preservation"] = {arm: dict(general=text_out.get(arm, {}).get("general"),
                                     kept_correct_pictures=float(np.mean([np.mean([arms_text[arm][s].get(i, False)
                                                                                   for s in arms_text[arm]])
                                                                          for i in base_ok])) if base_ok else None,
                                     n_base_correct=len(base_ok))
                           for arm in sorted(arms)}

    # ---------------------------------------------------------------- direct generation
    gans = answers(f"cells_{gen_set}_subject")
    gen, gen_choice, gen_text = {}, {}, {}
    for folder, recs in records(gen_dir, model).items():
        per = defaultdict(list)
        c = Counter()
        for r in recs:
            a = gans.get(r["file"], {})
            if len(a) < len(PANEL):
                continue
            chosen = [x["chosen"] for x in a.values()]
            per[r["item"]].append(sum(x == r["truth"] for x in chosen) >= 2)
            top, n = Counter(chosen).most_common(1)[0]
            c["referent" if top == r["truth"] and n >= 2 else "other thing" if n >= 2 and top == a[PANEL[0]]["literal"]
              else top.split(",")[0] if n >= 2 else "no majority"] += 1
        t = {i: all(v) for i, v in per.items() if len(v) == 2}
        if len(t) < sum(group_of[i] in GROUPS for i in range(len(reads))):
            print(f"  generation {folder}: {len(t)} texts judged so far, left out until complete")
            continue
        gen_text[folder] = t
        u = unit_success(t)
        gen[folder] = {g: float(np.mean([u[k] for k in units[g] if k in u])) for g in GROUPS}
        gen[folder]["tests"] = float(np.mean([u[k] for k in test_units if k in u]))
        gen[folder]["texts_tests"] = float(np.mean([ok for i, ok in t.items() if group_of[i] in TESTS]))
        gen_choice[folder] = dict(c)
    def gen_compare(f1, f2):
        shared = [i for i in range(len(reads)) if group_of[i] in TESTS and i in gen_text[f1] and i in gen_text[f2]]
        a = [float(gen_text[f1][i]) for i in shared]
        b = [float(gen_text[f2][i]) for i in shared]
        return dict(boot_diff(a, b, [pair_of(unit_of[i]) for i in shared]), n_texts=len(shared))

    for folder in gen:
        if folder != base and base in gen_text:
            gen[folder]["texts_vs_0"] = gen_compare(folder, base)
    if f"{model}_V_s0" in gen_text and f"{model}_R_s0" in gen_text:
        out["generation_V_vs_R_seed0"] = gen_compare(f"{model}_V_s0", f"{model}_R_s0")

    # pooled over training seeds: per text the mean over the arm's complete seed folders
    def gen_arm(arm):
        fs = [f for f in gen_text if arm_of(f)[0] == arm]
        shared = set.intersection(*(set(gen_text[f]) for f in fs)) if fs else set()
        return {i: float(np.mean([gen_text[f][i] for f in fs])) for i in shared}, len(fs)
    pooled = {}
    both_by_arm = {}
    for arm in sorted({arm_of(f)[0] for f in gen_text}):
        t, n = gen_arm(arm)
        u = defaultdict(list)
        for i, v in t.items():
            u[unit_of[i]].append(v)
        both = {k: float(np.prod(v)) for k, v in u.items() if len(v) == 2}  # chance both texts right, seeds independent
        both_by_arm[arm] = both
        pooled[arm] = dict(seeds=n, texts_tests=float(np.mean([v for i, v in t.items() if group_of[i] in TESTS])),
                           units_both_right=float(np.mean([both[k] for k in test_units if k in both])),
                           texts_by_group={g: float(np.mean([v for i, v in t.items() if group_of[i] == g])) for g in GROUPS})
    for a, b in (("G", "V"), ("G", "0"), ("V", "0")):
        if a in pooled and b in pooled:
            ta, tb = gen_arm(a)[0], gen_arm(b)[0]
            shared = [i for i in sorted(set(ta) & set(tb)) if group_of[i] in TESTS]
            pooled[f"{a}_vs_{b}"] = dict(boot_diff([ta[i] for i in shared], [tb[i] for i in shared],
                                                   [pair_of(unit_of[i]) for i in shared]), n_texts=len(shared))
            for g in TESTS:  # per test group: new pairs and forms, new verbs, held-out noun labels
                sg = [i for i in shared if group_of[i] == g]
                pooled[f"{a}_vs_{b}"][g] = boot_diff([ta[i] for i in sg], [tb[i] for i in sg], [pair_of(unit_of[i]) for i in sg])
    # pair level (ordinary drawing for both texts of a pair, as the lens measure): the difference in pair success over the
    # test units both arms have, noun pairs resampled together, own random stream
    pgen = np.random.default_rng(20261109)
    for a, b in (("G", "V"), ("G", "0"), ("V", "0")):
        if a in both_by_arm and b in both_by_arm:
            ks = [k for k in test_units if k in both_by_arm[a] and k in both_by_arm[b]]
            pooled[f"{a}_vs_{b}_units"] = dict(boot_diff_with(pgen, [both_by_arm[a][k] for k in ks],
                                                              [both_by_arm[b][k] for k in ks], [pair_of(k) for k in ks]),
                                               n_units=len(ks))
    out["generation_pooled"] = pooled
    out["generation"] = gen
    out["generation_choices"] = gen_choice

    PILOT.mkdir(parents=True, exist_ok=True)
    (PILOT / ("scores.json" if model == "bagel" else f"scores_{model}.json")).write_text(json.dumps(out, indent=1))
    print("LENS PICTURES, unit success (both referents of a pair drawn), mean over training seeds")
    for arm, r in lens.items():
        extra = "" if arm == "0" else (f"  vs 0: {r['vs_0_tests']['diff']:+.3f} [{r['vs_0_tests']['ci95'][0]:+.3f}, "
                                       f"{r['vs_0_tests']['ci95'][1]:+.3f}]  test3 {r['vs_0_test3']['diff']:+.3f}")
        print(f"  {arm}: dev {r.get('dev', float('nan')):.3f} test1 {r['test1']:.3f} test2 {r['test2']:.3f} test3 {r['test3']:.3f} "
              f"tests {r['tests']:.3f} (texts {r['texts_tests']:.3f}){extra}")
    for k in ("lens_V_vs_R_tests", "lens_V_vs_T_tests", "lens_G_vs_V_tests"):
        if k in out:
            print(f"  {k}: {out[k]['diff']:+.3f} [{out[k]['ci95'][0]:+.3f}, {out[k]['ci95'][1]:+.3f}]")
    print("TEXT (test groups, unit means)")
    for arm, r in text_out.items():
        extra = "" if arm == "0" else f"  untrained vs 0: {r['untrained_vs_0']['diff']:+.3f} [{r['untrained_vs_0']['ci95'][0]:+.3f}, {r['untrained_vs_0']['ci95'][1]:+.3f}]"
        print(f"  {arm}: name {r['name']:.3f} choice {r['choice']:.3f} yes {r['yes']:.3f} no {r['no']:.3f} "
              f"untrained {r['untrained']:.3f} (test3 {r['untrained_test3']:.3f}) general {r['general']:.3f}{extra}")
    print("PRESERVATION", json.dumps(out["preservation"]))
    print("DIRECT GENERATION (texts: both seeds show the referent alone)")
    for f, r in gen.items():
        extra = f"  texts vs 0: {r['texts_vs_0']['diff']:+.3f} [{r['texts_vs_0']['ci95'][0]:+.3f}, {r['texts_vs_0']['ci95'][1]:+.3f}]" if "texts_vs_0" in r else ""
        print(f"  {f}: texts {r['texts_tests']:.3f} units {r['tests']:.3f} dev {r.get('dev', float('nan')):.3f}{extra}  {gen_choice[f]}")
    print("DIRECT GENERATION POOLED OVER TRAINING SEEDS", json.dumps(out["generation_pooled"]))
    print("LENS CHOICES", json.dumps(out["lens_choices"]))


if __name__ == "__main__":
    main()
