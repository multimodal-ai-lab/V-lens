"""Reading two states (Supp. E: Table S12 and the two-state figure): does a drawing request with two slots, "a photo
of a plain thing", filled with two extracted states, show an attribute that one slot never shows? Items in
configs/prompts/two_slots.yaml (8 development and 8 held-out object pairs with two colour or pattern values each).

draw (GPU, sharded): states are taken at the model's insertion layer R and put unchanged into the slots at the same
layer R. Requests: one slot "a photo of a thing"; two slots "a photo of a plain thing" ("plain" takes the modifier,
"thing" the object; SenseNova-U1: "thing" and "plain thing"); alternative template "a photo of a thing that is plain".
Per pair and value, seeds 0 to 2:
  drawn        the explicit request drawn as usual ("a photo of a red car")
  one_noun     one slot: "car" from the explicit request
  two_native   two slots: "red" and "car" from the explicit request (the positive control)
  two_plain    two slots: "red" and "car" from a plain sentence
  two_context  two slots: "red" and "it" from "The car was painted bright red. Later I looked at it."
  one_it       one slot: "it" alone;  two_it_it  two slots: "it" twice
  two_cross    two slots: the OTHER value's modifier with this text's "it" (the colour should follow the modifier)
  one_mean     one slot: the average of "red" and "car" from the explicit request
  check_native, check_context   two_native and two_context in the template "a photo of a thing that is plain"
  cross_object two slots: this text's modifier with "it" from the partner object's text with the same value (the object
               should follow "it")
judge (GPU): attribute question for all but cross_object (values, "both or unclear", "something else", "there is no
<object>"); main-subject question between the two objects for cross_object.
score (CPU): a picture counts when two of three judges give the intended value, a state when two of three seeds do, a
pair when both values do; pairs count when both explicit requests are drawn right. Paired bootstrap of the context
pair against "it" alone. The object-required counts of Table S12 come from scripts/two_states_presence.py.

    python scripts/two_states.py draw --model show_o2 --shard 0 --of 4
    python scripts/two_states.py judge --model show_o2 --judge qwen3vl   (and gemma3, llavaov2)
    python scripts/two_states.py score
    python scripts/two_states.py score --models hunyuan --out scores_hunyuan.json
"""

import argparse
import json
import random
import time
import zlib
from collections import Counter, defaultdict

import numpy as np
import torch
import yaml

from vlens import prompts, readout, results
from vlens.carry import CopyTransform
from vlens.hidden import find_word, replace_at

OUT = results.RESULTS / "two_slots"
SEEDS = (0, 1, 2)
PANEL = ("qwen3vl", "gemma3", "llavaov2")
BOTH, ELSE = "both or unclear", "something else"
REQ = {"sensenova": dict(one="thing", two="plain thing", check="thing that is plain", named="{} {}")}
DEFAULT = dict(one="a photo of a thing", two="a photo of a plain thing", check="a photo of a thing that is plain",
               named="a photo of a {} {}")

def cfg():
    return yaml.safe_load((prompts.PROMPTS / "two_slots.yaml").read_text())


def label(c, v):
    return c["labels"].get(v, v)


def step_draw(args):
    from vlens.models import load
    c = cfg()
    pairs = {p["id"]: p for p in c["pairs"]}
    m = load(args.model)
    R = CopyTransform(args.model).R
    req = REQ.get(args.model, DEFAULT)
    target = {k: m.read_prompt(req[k]) for k in ("one", "two", "check")}
    slot = dict(one=[find_word(target["one"], "thing")[-1]],
                two=[find_word(target["two"], "plain")[-1], find_word(target["two"], "thing")[-1]],
                check=[find_word(target["check"], "plain")[-1], find_word(target["check"], "thing")[-1]])
    out = OUT / args.model
    out.mkdir(parents=True, exist_ok=True)
    records, jobs, t0 = [], [], time.time()
    tag = f"{args.shard}of{args.of}"

    def read_states(p, v):
        """States at layer R: modifier and noun of the explicit request and of the plain sentence; modifier and "it" of
        the context sentence."""
        ex = m.read_prompt(req["named"].format(v, p["object"]))
        pl = m.read(p["plain"].format(m=v))
        cx = m.read(p["context"].format(m=v))
        h = lambda r, w: r.hidden[R][readout.last_word(r, w)].float()  # noqa: E731
        return dict(ex_mod=h(ex, v), ex_noun=h(ex, p["object"]), pl_mod=h(pl, v), pl_noun=h(pl, p["object"]),
                    cx_mod=h(cx, v), cx_it=h(cx, "it"))

    def reading(kind, states):
        """kind one: [object]; two: [modifier, object] into "plain", "thing"; check: the same into the other template."""
        vecs = torch.stack([s.to(m.device) for s in states])
        return m.read_prompt(req[kind], edit=replace_at(R, slot[kind], vecs))

    def add(f, rd, **rec):
        for seed in SEEDS:
            name = f"{f}_s{seed}.png"
            jobs.append((name, rd, seed))
            records.append(dict(file=f"{args.model}/{name}", model=args.model, seed=seed, **rec))

    units = [(pid, j) for pid in pairs for j in range(2)]
    for pid, j in units[args.shard::args.of]:
        p = pairs[pid]
        v, o = p["values"][j], p["values"][1 - j]
        st, other = read_states(p, v), read_states(p, o)
        partner = read_states(pairs[p["partner"]], v)
        base = dict(pair=pid, split=p["split"], type=p["type"], object=p["object"], value=v, other=o,
                    question=p["question"], labels=[label(c, x) for x in p["values"]], truth=label(c, v),
                    alt=label(c, o), kind="value")
        add(f"{pid}_{v}_drawn", m.read_prompt(req["named"].format(v, p["object"])), way="drawn", **base)
        add(f"{pid}_{v}_one_noun", reading("one", [st["ex_noun"]]), way="one_noun", **base)
        add(f"{pid}_{v}_two_native", reading("two", [st["ex_mod"], st["ex_noun"]]), way="two_native", **base)
        add(f"{pid}_{v}_two_plain", reading("two", [st["pl_mod"], st["pl_noun"]]), way="two_plain", **base)
        add(f"{pid}_{v}_two_context", reading("two", [st["cx_mod"], st["cx_it"]]), way="two_context", **base)
        add(f"{pid}_{v}_one_it", reading("one", [st["cx_it"]]), way="one_it", **base)
        add(f"{pid}_{v}_two_it_it", reading("two", [st["cx_it"], st["cx_it"]]), way="two_it_it", **base)
        add(f"{pid}_{v}_one_mean", reading("one", [(st["ex_mod"] + st["ex_noun"]) / 2]), way="one_mean", **base)
        add(f"{pid}_{v}_check_native", reading("check", [st["ex_mod"], st["ex_noun"]]), way="check_native", **base)
        add(f"{pid}_{v}_check_context", reading("check", [st["cx_mod"], st["cx_it"]]), way="check_context", **base)
        # the other value's modifier with this text's "it": the attribute should follow the modifier
        add(f"{pid}_{v}_two_cross", reading("two", [other["cx_mod"], st["cx_it"]]), way="two_cross",
            **dict(base, truth=label(c, o), alt=label(c, v)))
        # this text's modifier with the partner object's "it" (same value): the object should follow "it"
        add(f"{pid}_{v}_cross_object", reading("two", [st["cx_mod"], partner["cx_it"]]), way="cross_object",
            **dict(base, kind="object", truth=pairs[p["partner"]]["object"], alt=p["object"]))
        readout.draw_all(m, jobs, out, args.batch)
        jobs.clear()
        torch.cuda.empty_cache()
        (out / f"records_{tag}.json").write_text(json.dumps(records, indent=1))
        print(f"[{time.time() - t0:6.0f}s] {pid} {v}", flush=True)


def options_for(r):
    if r["kind"] == "object":
        head = [r["truth"], r["alt"]]
        random.Random(r["file"]).shuffle(head)
        return head + ["both, as main subjects", "neither, as a main subject"]
    head = list(r["labels"])
    random.Random(r["file"]).shuffle(head)
    return head + [BOTH, ELSE, f"there is no {r['object']}"]


def question_for(r):
    q = ("Which of these is depicted as a main subject of the picture?" if r["kind"] == "object" else r["question"])

    def ask(opts):
        lines = [f"{chr(65 + i)}) {o}" for i, o in enumerate(opts)]
        return q + "\n" + "\n".join(lines) + "\nAnswer with the letter only."
    return ask


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
        ans.append(dict(file=r["file"], chosen=j.choose(Image.open(OUT / r["file"]), options_for(r), question_for(r)).chosen))
        if n % 200 == 0:
            f.write_text(json.dumps(ans, indent=1))
    f.write_text(json.dumps(ans, indent=1))


def step_score(args):
    summary = {}
    rng = np.random.default_rng(20261022)
    models = args.models.split(",") if args.models else ("bagel", "show_o2", "sensenova")
    for model in [x for x in models if (OUT / x).exists()]:
        votes = defaultdict(dict)
        for jn in PANEL:
            for f in OUT.glob(f"judge_{jn}_{model}_*.json"):
                for d in json.loads(f.read_text()):
                    votes[d["file"]][jn] = d["chosen"]
        cells = defaultdict(list)
        for r in records(model):
            v = votes.get(r["file"], {})
            if len(v) < len(PANEL):
                continue
            top, n = Counter(v.values()).most_common(1)[0]
            cells[(r["pair"], r["value"], r["way"], r["split"])].append(n >= 2 and top == r["truth"])
        ok = {k: sum(v) >= 2 for k, v in cells.items() if len(v) == 3}
        pairs = sorted({(k[0], k[3]) for k in ok})
        # eligible: both values drawn right from the explicit request
        elig = [(p, s) for p, s in pairs if all(ok.get((p, v, "drawn", s)) for v in {k[1] for k in ok if k[0] == p})]
        res = dict(eligible={s: sum(1 for _, ss in elig if ss == s) for s in ("dev", "test")}, ways={})
        ways = sorted({k[2] for k in ok})
        per = {}
        for w in ways:
            for s in ("dev", "test"):
                vals = []
                for p, ss in elig:
                    if ss != s:
                        continue
                    cs = [ok.get((p, v, w, s)) for v in sorted({k[1] for k in ok if k[0] == p})]
                    if None in cs or len(cs) != 2:
                        continue
                    vals.append(float(all(cs)) if w != "cross_object" else sum(cs) / 2)
                per[(w, s)] = vals
                res["ways"][f"{w}|{s}"] = f"{sum(vals):g}/{len(vals)}"
        dev_native = per.get(("two_native", "dev"), [])
        dev_one = per.get(("one_noun", "dev"), [])
        res["native_passes"] = bool(dev_native) and sum(dev_native) >= 5 and sum(dev_native) > sum(dev_one)

        def paired(a, b, s="test"):
            pa, pb = per.get((a, s), []), per.get((b, s), [])
            if not pa or len(pa) != len(pb):
                return None
            d = np.array(pa) - np.array(pb)
            bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            return dict(diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(d))
        res["test_context_minus_it"] = paired("two_context", "one_it")
        res["test_context_minus_it_it"] = paired("two_context", "two_it_it")
        res["test_native_minus_one_noun"] = paired("two_native", "one_noun")
        summary[model] = res
    (OUT / args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["draw", "judge", "score"])
    ap.add_argument("--model", default="show_o2")
    ap.add_argument("--judge", choices=PANEL)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--batch", type=int, default=5, help="pictures per batch for models with draw_many")
    # HunyuanImage is scored by the same rules into its own file:  score --models hunyuan --out scores_hunyuan.json
    ap.add_argument("--models", default="", help="score these models instead of the three main ones")
    ap.add_argument("--out", default="scores.json")
    args = ap.parse_args()
    dict(draw=step_draw, judge=step_judge, score=step_score)[args.step](args)


if __name__ == "__main__":
    main()
