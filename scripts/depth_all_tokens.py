"""The described object at every question token (Sec. 4.3, Fig. 4 "best later token", Supp. D).

The questions, layers, seeds and judging of scripts/depth_fixed_positions.py, inferred condition only, counted questions
only (right one-word answer, object not named first). Drawn here: the V-lens picture of every token of the question
except the tokens of the describing word. The question mark ("end") is already drawn and judged by
scripts/depth_fixed_positions.py for BAGEL, Show-o2 and SenseNova-U1, so it is drawn here only for HunyuanImage. A
question counts as shown by some other token at a layer when at least one such token shows the object in two of three
seeds.

HunyuanImage (about 105 s a picture on four GPUs) in two stages with the same result: seeds 3 and 4 first, then seed 5
only where exactly one of them shows the object (two equal verdicts already decide two of three).

The tokens the model reads after the question (instruction and chat format) and the first answer token are added by
scripts/depth_through_answer.py, whose score step gives the final curves of Fig. 4.

    python scripts/depth_all_tokens.py draw --model show_o2 --shard 0 --of 16 --batch 8
    python scripts/run_judge.py --judge qwen3vl --set cells:bridge_alltokens --subject --model show_o2   (and gemma3, llavaov2)
    python scripts/depth_all_tokens.py score
    # HunyuanImage: draw --seeds 3,4 --tag a, judge, then draw --seeds 5 --split-only --tag b, judge, score
"""

import argparse
import json
import time
from collections import Counter, defaultdict

import torch

from vlens import readout, results
from vlens.hidden import find_word

OUT = results.RESULTS / "bridge_alltokens"
CONFIRM = results.RESULTS / "bridge_confirm"
JUDGES = results.RESULTS / "judges"
PANEL = ("qwen3vl", "gemma3", "llavaov2")

fixed = results.load_script("depth_fixed_positions")


def counted(model):
    """Item indices of the questions the fixed-position run counts (inferred condition)."""
    ans = {}
    for f in (CONFIRM / model).glob("answers_*.json"):
        for a in json.loads(f.read_text()):
            ans[(a["item"], a["condition"])] = a
    return sorted(i for (i, c), a in ans.items() if c == "inferred" and a["right"] and not a["bridge_before_answer"])


def votes(folders):
    v = defaultdict(dict)
    for folder in folders:
        for j in PANEL:
            for f in JUDGES.glob(f"answers_{j}_cells_{folder}_subject_*.json"):
                for a in json.loads(f.read_text()):
                    v[a["file"]][j] = a["chosen"]
    return v


def shows(v, truth):
    """Two of three judges choose the object alone; None until all three have judged."""
    if len(v) < 3:
        return None
    top, n = Counter(v.values()).most_common(1)[0]
    return n >= 2 and top == truth


def step_draw(args):
    from vlens.models import load
    qs = dict(fixed.items()[0])
    use = counted(args.model)
    m = load(args.model)
    lens = readout.Lens(m, args.model)
    hy = args.model == "hunyuan"
    L = list(range(0, m.n_layers, 3)) + [m.n_layers - 1] if hy else readout.layers(m.n_layers)
    seeds = [int(s) for s in args.seeds.split(",")]
    redo = None
    if args.split_only:   # stage 2: seed 5 only where seeds 3 and 4 disagree
        v = votes(["bridge_alltokens"])
        cells = defaultdict(dict)
        for f in (OUT / args.model).glob("records_*.json"):
            for r in json.loads(f.read_text()):
                cells[(r["item"], r["position"], r["layer"])][r["seed"]] = shows(v.get(r["file"], {}), r["truth"])
        redo = {k for k, s in cells.items() if s.get(3) is not None and s.get(4) is not None and s[3] != s[4]}
        print(len(redo), "cells where seeds 3 and 4 disagree", flush=True)
    out = OUT / args.model
    out.mkdir(parents=True, exist_ok=True)
    units = []
    for i in use:
        it = qs[i]
        src = m.read(it["text"], system=fixed.HUNYUAN_SYSTEM) if hy else m.read(it["text"])
        content = [q for q in range(len(src.token_ids)) if src.content[q]]
        desc = set(find_word(src, it["positions"]["desc"]))
        assert readout.last_word(src, it["positions"]["desc"]) in desc
        for p in content:
            if p in desc or (p == content[-1] and not hy):   # the question mark is in the fixed-position run
                continue
            units.append((i, it, src, p))
    tag = f"{args.tag}{args.shard}of{args.of}"
    records, jobs, t0 = [], [], time.time()

    def flush():
        readout.draw_all(m, jobs, out, args.batch)
        jobs.clear()
        torch.cuda.empty_cache()
        (out / f"records_{tag}.json").write_text(json.dumps(records, indent=1))

    for i, it, src, p in units[args.shard::args.of]:
        pos = f"tok{p:02d}"
        for l in L:
            if redo is not None and (i, pos, l) not in redo:
                continue
            r = lens.reading(src.hidden[l][p], l)
            for seed in seeds:
                f = f"{it['id']}_inferred_{pos}_L{l:02d}_s{seed}.png"
                jobs.append((f, r, seed))
                records.append(dict(file=f"{args.model}/{f}", model=args.model, items="bridges", kind="reads", item=i,
                                    family=it["family"], condition="inferred", position=pos, layer=l, way="lens",
                                    seed=seed, token=src.token_strs[p], truth=it["truth"], alt=it["alt"]))
        flush()
        print(f"[{time.time() - t0:6.0f}s] {it['id']} token {p} {src.token_strs[p]!r}", flush=True)
    flush()
    print(len(records), "pictures ->", out)


def step_score(args):
    """Per model and layer: questions shown by the describing word, by the best other token (any token of the question
    outside the describing word, the question mark included), by the best later token (after the describing word), by the
    token before the answer; and which tokens show it."""
    qs = dict(fixed.items()[0])
    v = votes(["bridge_alltokens", "bridge_confirm"])
    summary = {}
    for model in [x for x in ("bagel", "show_o2", "sensenova", "hunyuan") if (OUT / x).exists()]:
        use = set(counted(model))
        recs = {}
        for folder in (OUT, CONFIRM):
            for f in (folder / model).glob("records_*.json"):
                for r in json.loads(f.read_text()):
                    if r["kind"] == "reads" and r.get("condition") == "inferred" and r["item"] in use:
                        recs[r["file"]] = r
        cells = defaultdict(dict)
        for fn, r in recs.items():
            cells[(r["item"], r["position"], r["layer"])][r["seed"]] = (shows(v.get(fn, {}), r["truth"]), r["token"])
        ok, pending = {}, 0
        for k, s in cells.items():
            verdicts = [x[0] for x in s.values()]
            if verdicts.count(True) >= 2:
                ok[k] = True
            elif verdicts.count(False) >= 2:
                ok[k] = False
            else:
                pending += 1
        other = lambda pos: pos.startswith("tok") or pos == "end"  # noqa: E731
        # the identifying word's last token: the gap the excluded word leaves among the drawn token positions (or right
        # after them, when the word stands just before the question mark)
        toks = defaultdict(set)
        for (i, pos, l) in cells:
            if pos.startswith("tok"):
                toks[i].add(int(pos[3:]))
        last = {i: max([p for p in range(min(ix), max(ix)) if p not in ix] or [max(ix) + 1]) for i, ix in toks.items()}
        later = lambda i, pos: pos == "end" or (pos.startswith("tok") and int(pos[3:]) > last[i])  # noqa: E731
        layers = sorted({k[2] for k in ok})
        res = {"counted": len(use), "pending_cells": pending, "desc": {}, "best_other": {}, "best_later": {},
               "preanswer": {}, "tokens_showing": {}}
        for l in layers:
            res["desc"][l] = sum(ok.get((i, "desc", l), False) for i in use)
            res["preanswer"][l] = sum(ok.get((i, "preanswer", l), False) for i in use)
            hits = {i: [cells[k][next(iter(cells[k]))][1] for k in ok if k[0] == i and k[2] == l and other(k[1]) and ok[k]]
                    for i in use}
            res["best_other"][l] = sum(bool(h) for h in hits.values())
            # the best token after the identifying word, the question mark included
            res["best_later"][l] = sum(any(ok[k] for k in ok if k[0] == i and k[2] == l and later(i, k[1])) for i in use)
            res["tokens_showing"][l] = {qs[i]["id"]: h for i, h in hits.items() if h}
        summary[model] = res
        print(model, "counted", len(use), "pending", pending)
        for l in layers:
            print(f"  L{l:2d}  desc {res['desc'][l]:2d}  best other {res['best_other'][l]:2d}  later {res['best_later'][l]:2d}  "
                  f"pre-answer {res['preanswer'][l]:2d}  {res['tokens_showing'][l] or ''}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["draw", "score"])
    ap.add_argument("--model", default="show_o2", choices=["bagel", "show_o2", "sensenova", "hunyuan"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8, help="pictures per batch for models with draw_many")
    ap.add_argument("--seeds", default="3,4,5")
    ap.add_argument("--split-only", action="store_true", help="draw only cells whose seeds 3 and 4 disagree")
    ap.add_argument("--tag", default="", help="added to the records file's name (HunyuanImage: a, then b)")
    args = ap.parse_args()
    dict(draw=step_draw, score=step_score)[args.step](args)


if __name__ == "__main__":
    main()
