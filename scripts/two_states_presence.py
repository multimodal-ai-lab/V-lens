"""Object presence for the two-state pictures (Supp. E: Tables S12 and S13 count a pair only with the object present).

Every picture of scripts/two_states.py (matched layers, --set main) and scripts/two_states_layers.py (8 layers per model,
--set layers) gets a second question, "Is there a <object> in this picture?" (yes / no), from the same three judges. A
picture then counts only when two of three judges give the expected colour or pattern AND two of three say the object is
there; states (two of three seeds), pairs (both values) and paired bootstrap intervals as before. Pairs enter only when
both explicit requests are drawn right with the object.

    python scripts/two_states_presence.py judge --judge qwen3vl --model show_o2 --set main   (every judge, both sets)
    python scripts/two_states_presence.py judge --judge qwen3vl --model show_o2 --set layers --shard 0 --of 2
    python scripts/two_states_presence.py score
    python scripts/two_states_presence.py score --models hunyuan --out scores_hunyuan.json
"""

import argparse
import json
import time
import zlib
from collections import Counter, defaultdict

import numpy as np

from vlens import results

OUT = results.RESULTS / "object_presence"
PANEL = ("qwen3vl", "gemma3", "llavaov2")


two_states = results.load_script("two_states")
two_states_layers = results.load_script("two_states_layers")
SETS = {"main": (two_states.OUT, two_states.records), "layers": (two_states_layers.OUT, two_states_layers.records)}


def question(obj):
    return f"Is there a {obj} in this picture?\nA) yes\nB) no\nAnswer with the letter only."


def step_judge(args):
    from PIL import Image
    from vlens.judges import Judge
    folder, recs = SETS[args.set]
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in OUT.glob(f"presence_{args.judge}_{args.set}_{args.model}_*.json"):
        done |= {d["file"] for d in json.loads(f.read_text())}
    todo = [r for r in recs(args.model) if r["file"] not in done and (folder / r["file"]).exists()
            and r.get("kind", "value") == "value" and zlib.crc32(r["file"].encode()) % args.of == args.shard]
    print(f"{len(todo)} pictures to judge", flush=True)
    if not todo:
        return
    j = Judge(args.judge)
    f = OUT / f"presence_{args.judge}_{args.set}_{args.model}_{args.shard}of{args.of}_{time.strftime('%H%M%S')}.json"
    ans = []
    for n, r in enumerate(todo):
        ans.append(dict(file=r["file"], chosen=j.choose(Image.open(folder / r["file"]), ["yes", "no"],
                                                        lambda opts, o=r["object"]: question(o)).chosen))
        if n % 200 == 0:
            f.write_text(json.dumps(ans, indent=1))
    f.write_text(json.dumps(ans, indent=1))


def picture_ok(folder, set_name, model, recs):
    """file -> (attribute right by two of three judges, object present by two of three)."""
    attr, pres = defaultdict(dict), defaultdict(dict)
    for jn in PANEL:
        for f in folder.glob(f"judge_{jn}_{model}_*.json"):
            for d in json.loads(f.read_text()):
                attr[d["file"]][jn] = d["chosen"]
        for f in OUT.glob(f"presence_{jn}_{set_name}_{model}_*.json"):
            for d in json.loads(f.read_text()):
                pres[d["file"]][jn] = d["chosen"]
    out = {}
    for r in recs:
        a, p = attr.get(r["file"], {}), pres.get(r["file"], {})
        if len(a) < 3 or len(p) < 3:
            continue
        top, n = Counter(a.values()).most_common(1)[0]
        out[r["file"]] = (n >= 2 and top == r["truth"], sum(v == "yes" for v in p.values()) >= 2)
    # say how many judged-kind pictures are left out because an attribute or presence answer is missing
    values = [r for r in recs if r.get("kind", "value") == "value"]
    if len(values) > len([r for r in values if r["file"] in out]):
        print(f"WARNING {model} {set_name}: {len(values) - len([r for r in values if r['file'] in out])} of "
              f"{len(values)} pictures lack an attribute or presence answer from one of the three judges and are "
              f"left out", flush=True)
    return out


def step_score(args):
    summary = {}
    rng = np.random.default_rng(20261023)
    cfg = two_states.cfg()
    values = {p["id"]: p["values"] for p in cfg["pairs"]}
    split = {p["id"]: p["split"] for p in cfg["pairs"]}
    for model in (args.models.split(",") if args.models else ("bagel", "show_o2", "sensenova")):
        res = {}
        main_recs = two_states.records(model)
        ok_main = picture_ok(two_states.OUT, "main", model, main_recs)
        # eligibility: both values drawn right AND with the object, from the explicit request (two of three seeds)
        cells = defaultdict(list)
        for r in main_recs:
            if r["file"] in ok_main and r["way"] == "drawn":
                cells[(r["pair"], r["value"])].append(all(ok_main[r["file"]]))
        elig = sorted(p for p in values if all(sum(cells.get((p, v), [])) >= 2 for v in values[p]))
        res["eligible"] = elig

        def pairs_right(recs, ok, key, both=True):
            c = defaultdict(list)
            for r in recs:
                if r["file"] in ok and r["pair"] in elig:
                    good = all(ok[r["file"]]) if both else ok[r["file"]][0]
                    c[key(r)].append(good)
            return {k: sum(v) >= 2 for k, v in c.items() if len(v) == 3}

        for set_name, recs_fn, folder, key in (
                ("main", two_states.records, two_states.OUT, lambda r: (r["pair"], r["value"], r["way"], None, None)),
                ("layers", two_states_layers.records, two_states_layers.OUT, lambda r: (r["pair"], r["value"], r["way"], r["layer"], r["readout"]))):
            recs = recs_fn(model)
            ok = ok_main if set_name == "main" else picture_ok(folder, "layers", model, recs)
            with_obj, colour_only = pairs_right(recs, ok, key), pairs_right(recs, ok, key, both=False)
            table = {}
            for (pair, value, way, layer, readout) in sorted({k for k in with_obj}, key=str):
                name = f"{way}|{readout}|{layer}"
                if name in table:
                    continue
                per_pair, per_pair_c = [], []
                for p in elig:
                    cs = [with_obj.get((p, v, way, layer, readout)) for v in values[p]]
                    cc = [colour_only.get((p, v, way, layer, readout)) for v in values[p]]
                    if None not in cs:
                        per_pair.append((float(all(cs)), split[p]))
                        per_pair_c.append(float(all(cc)))
                table[name] = dict(with_object=f"{sum(x for x, _ in per_pair):g}/{len(per_pair)}",
                                   colour_only=f"{sum(per_pair_c):g}/{len(per_pair_c)}",
                                   test_with_object=f"{sum(x for x, s in per_pair if s == 'test'):g}/{sum(s == 'test' for _, s in per_pair)}")
            res[set_name] = table
            # the primary contrast of the matched-layer run, with the object required
            if set_name == "main":
                for a, b in (("two_context", "one_it"), ("two_context", "two_it_it")):
                    da = [float(all(with_obj.get((p, v, a, None, None)) for v in values[p])) for p in elig if split[p] == "test"]
                    db = [float(all(with_obj.get((p, v, b, None, None)) for v in values[p])) for p in elig if split[p] == "test"]
                    if da and len(da) == len(db):
                        d = np.array(da) - np.array(db)
                        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                        res[f"test_{a}_minus_{b}_with_object"] = dict(diff=float(d.mean()), n=len(d),
                                                                      ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        summary[model] = res
    (OUT / args.out).write_text(json.dumps(summary, indent=1))
    for m, r in summary.items():
        print("=====", m, "eligible", len(r["eligible"]))
        for k, v in r.items():
            if k.startswith("test_"):
                print(" ", k, v)
        for k in sorted(r.get("layers", {})):
            if k.split("|")[0] in ("two_context", "two_native", "one_it"):
                print("  layers", k, r["layers"][k])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["judge", "score"])
    ap.add_argument("--model", default="show_o2")
    ap.add_argument("--set", choices=list(SETS), default="layers")
    ap.add_argument("--judge", choices=PANEL)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    # HunyuanImage is scored by the same rules into its own file:  score --models hunyuan --out scores_hunyuan.json
    ap.add_argument("--models", default="", help="score these models instead of the three main ones")
    ap.add_argument("--out", default="scores.json")
    args = ap.parse_args()
    dict(judge=step_judge, score=step_score)[args.step](args)


if __name__ == "__main__":
    main()
