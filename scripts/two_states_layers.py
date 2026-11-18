"""Reading two states across layers (Supp. E, depth sweep: Table S13 and the every-layer examples).

Same items, requests, seeds and judges as scripts/two_states.py, which reads every state at the insertion layer R.
Here the states are read at layer k, for 8 layers per model (spanning each model's depth and including R), and put into
the slots in two ways:
  lens  the V-lens: each state coordinate-scaled from layer k to R and put into its slot at R (as every one-slot V-lens
        picture in the paper);
  same  each state put back unchanged into its slot at layer k itself (read at k, insert at k).
Ways: one_noun, two_native (Expl.), two_context (Text), one_it, two_cross. Whether each value is drawn right from the
explicit request (eligibility) comes from scripts/two_states.py's judged "drawn" pictures: the same request, seeds and
model. With --examples: every layer, seed 0, for four items (the two-state examples figure).

draw (GPU, sharded over (pair, value) units); judge (GPU, three judges); score (CPU). The object-required counts of
Table S13 come from scripts/two_states_presence.py --set layers.

    python scripts/two_states_layers.py draw --model sensenova --shard 0 --of 8 --batch 3
    python scripts/two_states_layers.py judge --model sensenova --judge qwen3vl   (and gemma3, llavaov2)
    python scripts/two_states_layers.py score
    python scripts/two_states_layers.py score --models hunyuan --out scores_hunyuan.json
"""

import argparse
import json
import time
import zlib
from collections import Counter, defaultdict

import numpy as np
import torch

from vlens import readout, results
from vlens.carry import CopyTransform
from vlens.hidden import find_word, replace_at

OUT = results.RESULTS / "two_slots_layers"
LAYERS = dict(bagel=[0, 4, 8, 12, 16, 20, 24, 27], show_o2=[0, 4, 8, 12, 16, 20, 24, 27],
              sensenova=[0, 6, 12, 18, 24, 30, 36, 41],
              # HunyuanImage, 32 layers: eight layers spread evenly from the first to the last (numpy.linspace(0, 31, 8)
              # rounded), which holds its insertion layer 9, as each list above holds its model's R (8, 27, 24)
              hunyuan=[0, 4, 9, 13, 18, 22, 27, 31])
WAYS = ("one_noun", "two_native", "two_context", "one_it", "two_cross")
READOUTS = ("lens", "same")
# --examples: every layer, seed 0, for a few items
EXAMPLES = [("car", "red"), ("balloon", "yellow"), ("shirt", "striped"), ("bus", "green")]


two_states = results.load_script("two_states")


def step_draw(args):
    from vlens.models import load
    c = two_states.cfg()
    pairs = {p["id"]: p for p in c["pairs"]}
    m = load(args.model)
    s = CopyTransform(args.model).s
    R, mu = s["R"], s["mean"].float()
    sigma = readout.all_sigma(m, s, args.model)
    req = two_states.REQ.get(args.model, two_states.DEFAULT)
    target = {k: m.read_prompt(req[k]) for k in ("one", "two")}
    slot = dict(one=[find_word(target["one"], "thing")[-1]],
                two=[find_word(target["two"], "plain")[-1], find_word(target["two"], "thing")[-1]])
    layers = [int(x) for x in args.layers.split(",")] if args.layers else LAYERS[args.model]
    if args.examples:
        layers = list(range(m.n_layers))
    seeds = two_states.SEEDS[:1] if args.test or args.examples else two_states.SEEDS
    folder = OUT.with_name(OUT.name + "_test") if args.test else OUT.with_name(OUT.name + "_examples") if args.examples \
        else OUT
    out = folder / args.model
    out.mkdir(parents=True, exist_ok=True)
    records, jobs, t0 = [], [], time.time()
    tag = f"{args.shard}of{args.of}"

    def scaled(h, k):
        z = (h.float().cpu() - mu[k]) * (sigma[R] / sigma[k])
        return mu[R] + z * (s["spread_by_layer"][R] / z.norm())

    def reading(kind, states, k, readout):
        """states: hidden states read at layer k, one per slot ("plain", then "thing")."""
        if readout == "lens":
            vecs, at = torch.stack([scaled(h, k) for h in states]), R
        else:
            vecs, at = torch.stack([h.float().cpu() for h in states]), k
        return m.read_prompt(req[kind], edit=replace_at(at, slot[kind], vecs.to(m.device)))

    units = [(pid, j) for pid in pairs for j in range(2)]
    if args.examples:
        units = [(pid, pairs[pid]["values"].index(v)) for pid, v in EXAMPLES]
    mine = units[:1] if args.test else units[args.shard::args.of]
    for pid, j in mine:
        p = pairs[pid]
        v, o = p["values"][j], p["values"][1 - j]
        ex = m.read_prompt(req["named"].format(v, p["object"]))
        cx = m.read(p["context"].format(m=v))
        cx_other = m.read(p["context"].format(m=o))
        pos = dict(ex_mod=(ex, readout.last_word(ex, v)), ex_noun=(ex, readout.last_word(ex, p["object"])),
                   cx_mod=(cx, readout.last_word(cx, v)), cx_it=(cx, readout.last_word(cx, "it")),
                   ot_mod=(cx_other, readout.last_word(cx_other, o)))
        base = dict(pair=pid, split=p["split"], type=p["type"], object=p["object"], value=v, other=o,
                    question=p["question"], labels=[two_states.label(c, x) for x in p["values"]], truth=two_states.label(c, v),
                    alt=two_states.label(c, o), kind="value")
        for k in layers:
            h = {name: r.hidden[k][i] for name, (r, i) in pos.items()}
            plan = dict(one_noun=("one", [h["ex_noun"]]), two_native=("two", [h["ex_mod"], h["ex_noun"]]),
                        two_context=("two", [h["cx_mod"], h["cx_it"]]), one_it=("one", [h["cx_it"]]),
                        two_cross=("two", [h["ot_mod"], h["cx_it"]]))
            for readout in READOUTS:
                for way in WAYS:
                    kind, states = plan[way]
                    rd = reading(kind, states, k, readout)
                    rec = dict(base, truth=base["alt"], alt=base["truth"]) if way == "two_cross" else base
                    for seed in seeds:
                        name = f"{pid}_{v}_{readout}_L{k:02d}_{way}_s{seed}.png"
                        jobs.append((name, rd, seed))
                        records.append(dict(file=f"{args.model}/{name}", model=args.model, seed=seed, layer=k,
                                            readout=readout, way=way, **rec))
            if len(jobs) >= 100 * len(seeds):  # bounded memory: SenseNova-U1 cannot hold hundreds of readings at once
                readout.draw_all(m, jobs, out, args.batch)
                jobs.clear()
                torch.cuda.empty_cache()
        readout.draw_all(m, jobs, out, args.batch)  # all layers of a unit at once: fuller batches
        jobs.clear()
        torch.cuda.empty_cache()
        (out / f"records_{tag}.json").write_text(json.dumps(records, indent=1))
        print(f"[{time.time() - t0:6.0f}s] {pid} {v}", flush=True)


def records(model):
    seen = {}
    for f in (OUT / model).glob("records_*.json"):
        for r in json.loads(f.read_text()):
            seen[r["file"]] = r
    return list(seen.values())


def step_judge(args):
    from PIL import Image
    from vlens.judges import Judge
    done = set()
    for f in OUT.glob(f"judge_{args.judge}_{args.model}_*.json"):
        done |= {d["file"] for d in json.loads(f.read_text())}
    todo = [r for r in records(args.model) if r["file"] not in done and (OUT / r["file"]).exists()
            and zlib.crc32(r["file"].encode()) % args.of == args.shard]
    print(f"{len(todo)} pictures to judge", flush=True)
    j = Judge(args.judge)
    f = OUT / f"judge_{args.judge}_{args.model}_{args.shard}of{args.of}_{time.strftime('%H%M%S')}.json"
    ans = []
    for n, r in enumerate(todo):
        ans.append(dict(file=r["file"], chosen=j.choose(Image.open(OUT / r["file"]), two_states.options_for(r),
                                                        two_states.question_for(r)).chosen))
        if n % 200 == 0:
            f.write_text(json.dumps(ans, indent=1))
    f.write_text(json.dumps(ans, indent=1))


def cell_ok(folder, model, recs):
    """(pair, value, way, layer, readout) -> right in at least two of three seeds (a picture: two of three judges)."""
    votes = defaultdict(dict)
    for jn in two_states.PANEL:
        for f in folder.glob(f"judge_{jn}_{model}_*.json"):
            for d in json.loads(f.read_text()):
                votes[d["file"]][jn] = d["chosen"]
    cells = defaultdict(list)
    missing = 0
    for r in recs:
        v = votes.get(r["file"], {})
        if len(v) < len(two_states.PANEL):
            missing += 1
            continue
        top, n = Counter(v.values()).most_common(1)[0]
        cells[(r["pair"], r["value"], r["way"], r.get("layer"), r.get("readout"))].append(n >= 2 and top == r["truth"])
    if missing:  # say how many pictures are left out because a judge's answer is missing
        print(f"WARNING {model} {folder.name}: {missing} of {len(recs)} pictures lack an answer from one of the three "
              f"judges and are left out", flush=True)
    return {k: sum(v) >= 2 for k, v in cells.items() if len(v) == 3}


def step_score(args):
    summary = {}
    rng = np.random.default_rng(20261022)
    c = two_states.cfg()
    split = {p["id"]: p["split"] for p in c["pairs"]}
    values = {p["id"]: p["values"] for p in c["pairs"]}
    models = args.models.split(",") if args.models else ("bagel", "show_o2", "sensenova")
    for model in [x for x in models if (OUT / x).exists()]:
        drawn = cell_ok(two_states.OUT, model, two_states.records(model))
        elig = sorted(p for p in values if all(drawn.get((p, v, "drawn", None, None)) for v in values[p]))
        ok = cell_ok(OUT, model, records(model))
        res = dict(eligible=dict(dev=sum(split[p] == "dev" for p in elig), test=sum(split[p] == "test" for p in elig)),
                   curves={}, context_minus_it={})
        for readout in READOUTS:
            for k in LAYERS[model]:
                per = {}
                for way in WAYS:
                    vals = []
                    for p in elig:
                        cs = [ok.get((p, v, way, k, readout)) for v in values[p]]
                        if None not in cs:
                            vals.append((float(all(cs)), split[p]))
                    per[way] = vals
                    res["curves"][f"{readout}|{k}|{way}"] = dict(
                        all=f"{sum(x for x, _ in vals):g}/{len(vals)}",
                        test=f"{sum(x for x, s in vals if s == 'test'):g}/{sum(s == 'test' for _, s in vals)}",
                        cells=sum(bool(ok.get((p, v, way, k, readout))) for p in elig for v in values[p]))
                a, b = [x for x, _ in per["two_context"]], [x for x, _ in per["one_it"]]
                if a and len(a) == len(b):
                    d = np.array(a) - np.array(b)
                    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                    res["context_minus_it"][f"{readout}|{k}"] = dict(
                        diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        n=len(d))
        summary[model] = res
    (OUT / args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["draw", "judge", "score"])
    ap.add_argument("--model", default="show_o2")
    ap.add_argument("--judge", choices=two_states.PANEL)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--layers", help="comma-separated, overrides LAYERS (test runs)")
    ap.add_argument("--test", action="store_true", help="one unit, seed 0, into a _test folder")
    ap.add_argument("--examples", action="store_true", help="EXAMPLES at every layer, seed 0, into an _examples folder")
    # HunyuanImage is scored by the same rules into its own file:  score --models hunyuan --out scores_hunyuan.json
    ap.add_argument("--models", default="", help="score these models instead of the three main ones")
    ap.add_argument("--out", default="scores.json")
    args = ap.parse_args()
    dict(draw=step_draw, judge=step_judge, score=step_score)[args.step](args)


if __name__ == "__main__":
    main()
