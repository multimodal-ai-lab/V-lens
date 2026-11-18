"""The described object across layers at three fixed positions (Sec. 4.3, Supp. D: Fig. S1, Tables S9 and S10).

Items: the 16 two-hop questions of configs/prompts/bridges.yaml (families "bridge" and "same_answer"), each as written
("inferred") and after a sentence that names the described object ("explicit", configs/prompts/bridge_confirm.yaml).

draw (GPU, sharded): per question and condition, the model's greedy one-word reply (the question plus " Answer with one
    word."; HunyuanImage with a one-word system prompt), then the V-lens picture (vlens.readout.Lens, the coordinate
    scaling of Eq. 1 into the model's insertion layer) of three positions: the describing word's last token ("desc"),
    the question's last token ("end") and the token right before the first right answer in the reply ("preanswer"),
    from every second layer (every third for models deeper than 30 layers) plus the last, seeds 3, 4 and 5. Named
    controls: each object's name in "Yesterday I saw a {name} on my way home." at the same layers and seeds.
    HunyuanImage: every third layer and the last, the describing word (inferred condition) and the token before the
    answer only; its other positions come from scripts/depth_all_tokens.py and the `spots` step.
spots (GPU): HunyuanImage's remaining curves of Fig. S1 (explicit condition, describing word and question end), seeds
    3 and 4 first, then seed 5 only where their verdicts differ (--split-only), which gives the same two-of-three result.
score (CPU): a picture shows the object when two of three judges choose it alone; a cell (question, position, layer)
    when two of three seeds do; a question counts in a condition when its one-word answer is right and does not name
    the object before the answer. Writes the curves, the named controls and the paired bootstrap comparisons of
    Table S10 (Show-o2, before the answer, layer 24 against 27, both conditions) to scores.json.

Pictures and records in $VLENS_OUT/bridge_confirm/<model>/. Judging (three judges, main-subject question):

    python scripts/depth_fixed_positions.py draw --model show_o2 --shard 0 --of 4
    python scripts/run_judge.py --judge qwen3vl --set cells:bridge_confirm --subject --model show_o2   (and gemma3, llavaov2)
    python scripts/depth_fixed_positions.py score
"""

import argparse
import json
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import yaml

from vlens import prompts, readout, results

OUT = results.RESULTS / "bridge_confirm"
JUDGES = results.RESULTS / "judges"
SEEDS = (3, 4, 5)
PRIMARY = dict(model="show_o2", position="preanswer", condition="inferred", layer=24, last=27)
# HunyuanImage answers in one word with this system prompt (as in scripts/state_questions.py)
HUNYUAN_SYSTEM = "You are a helpful assistant. Answer every question with exactly one word. Never explain your answer."


def items():
    reads = yaml.safe_load((prompts.PROMPTS / "bridges.yaml").read_text())
    intro = yaml.safe_load((prompts.PROMPTS / "bridge_confirm.yaml").read_text())["explicit_intro"]
    qs = [(i, it) for i, it in enumerate(reads["reads"]) if it["family"] in ("bridge", "same_answer")]
    truths = {it["truth"] for _, it in qs}
    ctl = [(j, c) for j, c in enumerate(reads["controls"]) if c["object"] in truths]
    return qs, ctl, intro


def step_draw(args):
    from vlens.models import load
    qs, ctl, intro = items()
    m = load(args.model)
    lens = readout.Lens(m, args.model)
    L = readout.layers(m.n_layers)
    hy = args.model == "hunyuan"
    if hy:   # HunyuanImage: every third layer and the last, only the positions the figure plots
        L = list(range(0, m.n_layers, 3)) + [m.n_layers - 1]
    out = OUT / args.model
    out.mkdir(parents=True, exist_ok=True)
    units = [("q", i, it, cond) for i, it in qs for cond in ("inferred", "explicit")] + [("c", j, c, None) for j, c in ctl]
    records, answers, jobs, t0 = [], [], [], time.time()
    tag = f"{args.shard}of{args.of}"

    def flush():
        readout.draw_all(m, jobs, out, args.batch)
        jobs.clear()
        torch.cuda.empty_cache()
        (out / f"records_{tag}.json").write_text(json.dumps(records, indent=1))
        (out / f"answers_{tag}.json").write_text(json.dumps(answers, indent=1))

    for kind, i, it, cond in units[args.shard::args.of]:
        if kind == "c":
            src = m.read(it["text"], system=HUNYUAN_SYSTEM) if hy else m.read(it["text"])
            p = readout.last_word(src, it["object"])
            for l in L:
                r = lens.reading(src.hidden[l][p], l)
                for seed in SEEDS:
                    f = f"ctl_{it['object'].replace(' ', '-')}_L{l:02d}_s{seed}.png"
                    jobs.append((f, r, seed))
                    records.append(dict(file=f"{args.model}/{f}", model=args.model, items="bridges", kind="controls",
                                        item=i, position="word", layer=l, way="lens", seed=seed, truth=it["object"],
                                        alt=it["literals"][0]))
            flush()
            continue
        text = it["text"] if cond == "inferred" else intro[it["id"]] + " " + it["text"]
        asked = text + readout.ONE_WORD
        reply = m.answer_one_word(asked, HUNYUAN_SYSTEM) if hy else m.answer(asked, max_new_tokens=readout.MAX_REPLY)
        hit = readout.answer_hit(reply, it["answers"])
        before = hit is not None and readout.answer_hit(reply[: hit[0]], readout.names(it)) is not None
        answers.append(dict(item=i, id=it["id"], condition=cond, reply=reply, right=hit is not None,
                            bridge_before_answer=before))
        src = m.read(text, system=HUNYUAN_SYSTEM) if hy else m.read(text)
        content = [q for q in range(len(src.token_ids)) if src.content[q]]
        spots = [("desc", src, readout.last_word(src, it["positions"]["desc"])), ("end", src, content[-1])]
        if hy:   # the describing word (inferred only) and the token before the answer
            spots = [sp for sp in spots if sp[0] == "desc" and cond == "inferred"]
        if hit is not None:
            rep = m.read_reply(asked, reply, system=HUNYUAN_SYSTEM) if hy else m.read_reply(asked, reply)
            rc = [q for q in range(len(rep.token_ids)) if rep.content[q]]
            starts, full = [], ""
            for q in rc:
                starts.append(len(full))
                full += rep.token_strs[q]
            off = len(full) - len(full.lstrip())
            k = [n for n, s0 in enumerate(starts) if s0 + len(rep.token_strs[rc[n]]) > hit[0] + off][0]
            spots.append(("preanswer", rep, rc[k] - 1))
        for pos, reading, p in spots:
            for l in L:
                r = lens.reading(reading.hidden[l][p], l)
                for seed in SEEDS:
                    f = f"{it['id']}_{cond}_{pos}_L{l:02d}_s{seed}.png"
                    jobs.append((f, r, seed))
                    records.append(dict(file=f"{args.model}/{f}", model=args.model, items="bridges", kind="reads", item=i,
                                        family=it["family"], condition=cond, position=pos, layer=l, way="lens",
                                        seed=seed, token=reading.token_strs[p], truth=it["truth"], alt=it["alt"]))
            flush()
        print(f"[{time.time() - t0:6.0f}s] {it['id']} {cond}: {reply!r} right={hit is not None}", flush=True)
    flush()


def all_votes(folders=("bridge_confirm",)):
    votes = defaultdict(dict)
    for folder in folders:
        for j in ("qwen3vl", "gemma3", "llavaov2"):
            for f in JUDGES.glob(f"answers_{j}_cells_{folder}_subject_*.json"):
                for a in json.loads(f.read_text()):
                    votes[a["file"]][j] = a["chosen"]
    return votes


ALLTOKENS = results.RESULTS / "bridge_alltokens"


def alltokens_end(model):
    """The inferred question end of a model whose run here skipped it (HunyuanImage): drawn and judged by
    scripts/depth_all_tokens.py as the question's last token (the question mark, the same position as "end" here: the
    last content token of the same reading), same layers and seeds. Returned as records with position "end"."""
    recs = {}
    for f in (ALLTOKENS / model).glob("records_*.json"):
        for r in json.loads(f.read_text()):
            recs[r["file"]] = r
    last = defaultdict(int)
    for r in recs.values():
        last[r["item"]] = max(last[r["item"]], int(r["position"][3:]))
    out = {fn: dict(r, position="end") for fn, r in recs.items() if int(r["position"][3:]) == last[r["item"]]}
    assert all(r["token"].strip() == "?" for r in out.values()), "the last drawn token is not the question mark"
    return out


def counted(model, cond):
    """Item indices of the questions the curves count in a condition: right one-word answer, object not named first."""
    ans = {}
    for f in (OUT / model).glob("answers_*.json"):
        for a in json.loads(f.read_text()):
            ans[(a["item"], a["condition"])] = a
    return sorted(i for (i, c), a in ans.items() if c == cond and a["right"] and not a["bridge_before_answer"])


def step_spots(args):
    """HunyuanImage's remaining curves of Fig. S1: --spots condition:position pairs (the explicit condition's
    describing word and question end), at step_draw's layers (every third and the last), for the questions the curves
    count in that condition (score() counts no other), no answers redrawn. In two stages: --seeds 3,4, then --seeds 5
    --split-only where the two verdicts differ, which gives the same two-of-three result. Records in
    records_spots<tag><shard>of<of>.json."""
    from vlens.models import load
    qs, _, intro = items()
    qs = dict(qs)
    want = [tuple(s.split(":")) for s in args.spots.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    redo = None
    if args.split_only:
        v = all_votes()
        cells = defaultdict(dict)
        for f in (OUT / args.model).glob("records_spots*.json"):
            for r in json.loads(f.read_text()):
                vv = v.get(r["file"], {})
                if len(vv) == 3:
                    top, n = Counter(vv.values()).most_common(1)[0]
                    cells[(r["item"], r["condition"], r["position"], r["layer"])][r["seed"]] = n >= 2 and top == r["truth"]
        redo = {k for k, s in cells.items() if 3 in s and 4 in s and s[3] != s[4]}
        print(len(redo), "cells where seeds 3 and 4 disagree", flush=True)
    m = load(args.model)
    lens = readout.Lens(m, args.model)
    hy = args.model == "hunyuan"
    L = list(range(0, m.n_layers, 3)) + [m.n_layers - 1] if hy else readout.layers(m.n_layers)
    out = OUT / args.model
    units = [(i, cond, pos) for cond in sorted({c for c, _ in want}) for i in counted(args.model, cond)
             for pos in [p for c, p in want if c == cond]]
    tag = f"spots{args.tag}{args.shard}of{args.of}"
    records, jobs, t0 = [], [], time.time()
    for i, cond, pos in units[args.shard::args.of]:
        it = qs[i]
        text = it["text"] if cond == "inferred" else intro[it["id"]] + " " + it["text"]
        src = m.read(text, system=HUNYUAN_SYSTEM) if hy else m.read(text)
        content = [q for q in range(len(src.token_ids)) if src.content[q]]
        p = readout.last_word(src, it["positions"]["desc"]) if pos == "desc" else content[-1]
        for l in L:
            if redo is not None and (i, cond, pos, l) not in redo:
                continue
            r = lens.reading(src.hidden[l][p], l)
            for seed in seeds:
                f = f"{it['id']}_{cond}_{pos}_L{l:02d}_s{seed}.png"
                jobs.append((f, r, seed))
                records.append(dict(file=f"{args.model}/{f}", model=args.model, items="bridges", kind="reads", item=i,
                                    family=it["family"], condition=cond, position=pos, layer=l, way="lens",
                                    seed=seed, token=src.token_strs[p], truth=it["truth"], alt=it["alt"]))
        readout.draw_all(m, jobs, out, args.batch)
        jobs.clear()
        torch.cuda.empty_cache()
        (out / f"records_{tag}.json").write_text(json.dumps(records, indent=1))
        print(f"[{time.time() - t0:6.0f}s] {it['id']} {cond} {pos} {src.token_strs[p]!r}", flush=True)
    (out / f"records_{tag}.json").write_text(json.dumps(records, indent=1))
    print(len(records), "pictures ->", out)


def step_score(args):
    votes = all_votes(("bridge_confirm", "bridge_alltokens"))
    qs, ctl, _ = items()
    rng = np.random.default_rng(20261022)
    summary = {}
    for model in [x for x in ("bagel", "show_o2", "sensenova", "hunyuan") if (OUT / x).exists()]:
        recs, ans = {}, {}
        for f in (OUT / model).glob("records_*.json"):
            for r in json.loads(f.read_text()):
                recs[r["file"]] = r
        if model == "hunyuan" and not any(r.get("condition") == "inferred" and r["position"] == "end" for r in recs.values()):
            recs.update(alltokens_end(model))   # HunyuanImage's question end, from scripts/depth_all_tokens.py
        for f in (OUT / model).glob("answers_*.json"):
            for a in json.loads(f.read_text()):
                ans[(a["item"], a["condition"])] = a
        cells = defaultdict(list)
        for fn, r in recs.items():
            v = votes.get(fn, {})
            if len(v) < 3:
                continue
            top, n = Counter(v.values()).most_common(1)[0]
            cells[(r["kind"], r["item"], r.get("condition"), r["position"], r["layer"])].append(n >= 2 and top == r["truth"])
        # two of three seeds; HunyuanImage's added curves draw seed 5 only where seeds 3 and 4 disagree (step_spots), so
        # two equal verdicts decide a cell there; a cell of two different verdicts waits for its third seed
        ok = {k: sum(v) >= 2 for k, v in cells.items() if len(v) == 3 or (len(v) == 2 and v[0] == v[1])}
        print(model, "cells waiting for a third seed:", sum(1 for v in cells.values() if len(v) == 2 and v[0] != v[1]))
        # a question counts in a condition when its one-word answer is right and does not name the object first
        use = {(i, c) for (i, c), a in ans.items() if a["right"] and not a["bridge_before_answer"]}
        L = sorted({k[4] for k in ok})
        res = dict(eligible={c: sum(1 for (i, cc) in use if cc == c) for c in ("inferred", "explicit")}, curves={})
        for cond in ("inferred", "explicit"):
            for pos in ("desc", "end", "preanswer"):
                row = {}
                for l in L:
                    v = [ok[("reads", i, cond, pos, l)] for i, _ in qs if (i, cond) in use and ("reads", i, cond, pos, l) in ok]
                    row[l] = f"{sum(v)}/{len(v)}"
                res["curves"][f"{cond}|{pos}"] = row
        res["controls"] = {l: f"{sum(ok[k] for k in ok if k[0] == 'controls' and k[4] == l)}/"
                              f"{sum(1 for k in ok if k[0] == 'controls' and k[4] == l)}" for l in L}

        def paired(cond_a, l_a, cond_b, l_b, pos):
            d = [float(ok[("reads", i, cond_a, pos, l_a)]) - float(ok[("reads", i, cond_b, pos, l_b)]) for i, _ in qs
                 if (i, cond_a) in use and (i, cond_b) in use and ("reads", i, cond_a, pos, l_a) in ok
                 and ("reads", i, cond_b, pos, l_b) in ok]
            if not d:
                return None
            d = np.array(d)
            bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            return dict(diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(d))
        if model == PRIMARY["model"]:
            l24, last = PRIMARY["layer"], PRIMARY["last"]
            res["primary_inferred_24_minus_last"] = paired("inferred", l24, "inferred", last, "preanswer")
            res["explicit_24_minus_last"] = paired("explicit", l24, "explicit", last, "preanswer")
            res["inferred_minus_explicit_at_24"] = paired("inferred", l24, "explicit", l24, "preanswer")
            res["inferred_minus_explicit_at_last"] = paired("inferred", last, "explicit", last, "preanswer")
        summary[model] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["draw", "spots", "score"])
    ap.add_argument("--model", default="show_o2", choices=["bagel", "show_o2", "sensenova", "hunyuan"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--batch", type=int, default=5, help="pictures per batch for models with draw_many")
    ap.add_argument("--spots", default="explicit:desc,explicit:end", help="spots: condition:position pairs")
    ap.add_argument("--seeds", default="3,4", help="spots: drawing seeds")
    ap.add_argument("--split-only", action="store_true", help="spots: only cells whose seeds 3 and 4 disagree")
    ap.add_argument("--tag", default="", help="spots: added to the records file's name (stage 2: b)")
    args = ap.parse_args()
    dict(draw=step_draw, spots=step_spots, score=step_score)[args.step](args)


if __name__ == "__main__":
    main()
