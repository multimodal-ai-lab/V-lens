"""Held thoughts while copying, the systematic search of Supp. J (extending the examples of Sec. 4.4,
scripts/held_thoughts.py). Items: configs/prompts/held_search.yaml.

The model copies one of 14 neutral sentences while thinking about one of twelve drawable things (Show-o2) or eight
(HunyuanImage-3.0): 'Write "{sentence}" exactly. Think about {thing} while you write. Don't write anything else.'
Controls: the same sentence without the middle sentence (no instruction), and the same sentence while thinking about the
thing's partner (swap). A reply counts only when its case-insensitive Ratcliff-Obershelp similarity to the sentence is at
least 0.85 (difflib). Read tokens: for every word and punctuation mark of the sentence, the token of the model's own copy
that covers its last character. Every read token of a kept copy is drawn through the V-lens (coordinate-scaled copy into
the insertion layer, "a photo of a thing", vlens.readout.Lens) at layers 24, 25 and 26 with drawing seeds 0, 1 and 2
(HunyuanImage: seed 2 only where seeds 0 and 1 disagree), and judged by the three judges (main subject among the pair's
two things, both, neither). A token and layer show a thing when at least two seeds do; the controls are scanned the
same way.

Steps (outputs in $VLENS_OUT/held_search/, hidden states in $VLENS_DATA/held_search/<model>/):
  reply   (GPU) greedy reply to every prompt, its similarity, the read tokens, and the hidden states entering every
          layer at the instruction's end and every reply token
  plan    (CPU) the pictures of the model's grid: kept trials, read tokens, layers, seeds -> plan_<model>.json
  draw    (GPU, sharded by a hash of the file name) planned pictures that are missing
  judge   (GPU, sharded) the three judges (--judge); a picture of a no-instruction trial is judged once for every pair
  adapt   (CPU, HunyuanImage) seed 2 wherever seeds 0 and 1 disagree on a scored verdict -> plan_<model>_r2.json
          (then draw and judge again)
  score   (CPU) per model: kept copies, successful token and layer combinations by layer and token kind, the controls,
          and every success's best cell with its matched controls -> scores_<model>.json

    python scripts/held_search.py reply --model show_o2
    python scripts/held_search.py plan --model show_o2
    python scripts/held_search.py draw --model show_o2 --batch 8 --shard 0 --of 8
    python scripts/held_search.py judge --model show_o2 --judge qwen3vl     (and gemma3, llavaov2)
    python scripts/held_search.py score --model show_o2
    # HunyuanImage (four GPUs for reply and draw): reply, plan, draw, judge, then adapt, draw, judge, score
"""

import argparse
import difflib
import json
import math
import os
import re
import time
import zlib
from collections import Counter, defaultdict

import yaml

from vlens import paths, prompts, results

OUT = results.RESULTS / "held_search"
STATES = paths.DATA / "held_search"
PANEL = ("qwen3vl", "gemma3", "llavaov2")
COPY_MIN = 0.85
TAG = re.compile(r"</?[a-z_]+>")   # HunyuanImage's control tags (</think>, <recaption>, ...), never part of the copy
UNIT = re.compile(r"[a-z']+|[.,;:!?]")
DETERMINERS = {"the", "a", "an"}


def write_json(path, obj):
    """Write atomically (draw and judge jobs read plan and answer files while other jobs write them)."""
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def cfg():
    return yaml.safe_load((prompts.PROMPTS / "held_search.yaml").read_text())


def slug(thing):
    return re.sub(r"^(a|an|the) ", "", thing.lower()).replace(" ", "_")


def sentences(c):
    return {s["id"]: s for s in c["sentences"]}


def trial_id(sid, thing=None):
    return f"{sid}_none" if thing is None else f"{sid}_think_{slug(thing)}"


def all_trials(c, model):
    """Every prompt of the config, the model's grid first: dicts with trial, sid, cond, pair, k, thing, prompt."""
    grid = c["models"][model]
    out = []
    for s in c["sentences"]:
        for p, pair in enumerate(c["pairs"]):
            for k, thing in enumerate(pair):
                out.append(dict(trial=trial_id(s["id"], thing), sid=s["id"], cond="think", pair=p, k=k, thing=thing,
                                prompt=c["templates"]["think"].format(sentence=s["text"], thing=thing),
                                grid=s["id"] in grid["sentences"] and p in grid["pairs"]))
        out.append(dict(trial=trial_id(s["id"]), sid=s["id"], cond="none", pair=None, k=None, thing=None,
                        prompt=c["templates"]["none"].format(sentence=s["text"]), grid=s["id"] in grid["sentences"]))
    return sorted(out, key=lambda t: not t["grid"])   # stable: the grid first, in config order


def similarity(reply, sentence):
    a = TAG.sub("", reply).strip().strip('"“”').strip().lower()
    return difflib.SequenceMatcher(None, a, sentence.strip().lower()).ratio()


def units(sentence):
    """The words and punctuation marks of a sentence, lower case."""
    return UNIT.findall(sentence.lower())


def unit_positions(strs, sentence):
    """For every unit of the sentence, the index into strs (the copy's tokens; the caller adds 1) of the token covering the
    unit's last character in the reply, or None where the reply lacks that unit. The reply's text is aligned to the
    sentence with difflib's matching blocks (case-insensitive), so text before or after the copy is never read."""
    starts, full = [], ""
    for s in strs:
        starts.append(len(full))
        full += s
    sent = sentence.lower()
    sm = difflib.SequenceMatcher(None, full.lower(), sent, autojunk=False)
    char = {}
    for a, b, n in sm.get_matching_blocks():
        for i in range(n):
            char[b + i] = a + i
    out = []
    for m in UNIT.finditer(sent):   # the whole unit must be copied, in one piece
        if not all(j in char for j in range(m.start(), m.end())) or char[m.end() - 1] - char[m.start()] != m.end() - m.start() - 1:
            out.append(None)
            continue
        end = char[m.end() - 1]
        out.append(max(i for i, s0 in enumerate(starts) if s0 <= end))
    return out


def kind(unit):
    return "determiner" if unit in DETERMINERS else ("punctuation" if not unit[0].isalpha() else "word")


# ------------------------------------------------------------------------------------------------------------ reply
def instruction_end(reading, first_reply):
    """The prompt's last text token: the full stop of "Don't write anything else." (or "else" itself)."""
    q = max(i for i in range(first_reply) if "else" in reading.token_strs[i])
    return q + 1 if q + 1 < first_reply and reading.token_strs[q + 1].strip() == "." else q


def step_reply(args):
    import torch
    from vlens.models import load
    c = cfg()
    todo = all_trials(c, args.model)
    if args.grid_only:
        todo = [t for t in todo if t["grid"]]
    todo = todo[args.shard::args.of]
    results.folder("held_search")
    (STATES / args.model).mkdir(parents=True, exist_ok=True)
    f = OUT / f"replies_{args.model}_{args.shard}of{args.of}.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    done = {r["trial"] for r in rows}
    todo = [t for t in todo if t["trial"] not in done]
    print(len(todo), "replies to make", flush=True)
    if not todo:
        return
    m = load(args.model)
    sents, t0 = sentences(c), time.time()
    for t in todo:
        text, sentence = t["prompt"], sents[t["sid"]]["text"]
        if args.model == "hunyuan":   # the closing tag is barred from the first position (an empty reply otherwise)
            reply, reading = m._follow(text, None, args.max_new_tokens, require_word=True)
        else:
            reply = m.answer(text, max_new_tokens=args.max_new_tokens)
            reading = m.read_reply(text, reply)
        reply_pos = [q for q in range(len(reading.token_ids))
                     if reading.content[q] and not TAG.fullmatch(reading.token_strs[q].strip())]
        row = dict(t, reply=reply)
        if not reply_pos:
            row.update(similarity=0.0, kept=False, tokens=[], units=[])
        else:
            positions = [instruction_end(reading, reply_pos[0])] + reply_pos
            strs = [reading.token_strs[q] for q in positions]
            hidden = torch.stack([torch.stack([reading.hidden[l][q] for q in positions]) for l in range(m.n_layers)])
            torch.save(dict(positions=positions, hidden=hidden.half()), STATES / args.model / f"{t['trial']}.pt")
            sim = similarity(reply, sentence)
            row.update(similarity=round(sim, 4), kept=sim >= COPY_MIN, tokens=strs,
                       units=[None if u is None else u + 1 for u in unit_positions(strs[1:], sentence)])
        rows.append(row)
        f.write_text(json.dumps(rows, indent=1))
        print(f"[{time.time() - t0:6.0f}s] {t['trial']}: {row['similarity']:.2f} {reply!r}", flush=True)


def replies(model):
    rows = {}
    for f in sorted(OUT.glob(f"replies_{model}_*.json")):
        for r in json.loads(f.read_text()):
            rows[r["trial"]] = r
    return rows


# ------------------------------------------------------------------------------------------------------------- plan
def picture(model, trial, pos, layer, seed):
    return f"{model}/{trial}_t{pos:02d}_L{layer:02d}_s{seed}.png"


def judge_items(e, c, model):
    """[(judge id, [thing, partner order as in the config])] of a plan entry."""
    if e["cond"] == "think":
        return [(e["file"], c["pairs"][e["pair"]])]
    return [(f"{e['file']}|p{p}", c["pairs"][p]) for p in c["models"][model]["pairs"]]


def step_plan(args):
    c, rep = cfg(), replies(args.model)
    grid, sents = c["models"][args.model], sentences(c)
    want = [t for t in all_trials(c, args.model) if t["grid"]]
    missing = [t["trial"] for t in want if t["trial"] not in rep]
    if missing:
        raise SystemExit(f"{len(missing)} grid replies missing, e.g. {missing[:5]}")
    plan = []
    for t in want:
        r = rep[t["trial"]]
        if not r["kept"]:
            continue
        us = units(sents[t["sid"]]["text"])
        by_pos = defaultdict(list)
        for u, pos in enumerate(r["units"]):
            if pos is not None:
                by_pos[pos].append(u)
        for pos, uu in sorted(by_pos.items()):
            for layer in grid["layers"]:
                for seed in grid["seeds"]:
                    plan.append(dict(file=picture(args.model, t["trial"], pos, layer, seed), model=args.model,
                                     trial=t["trial"], sid=t["sid"], cond=t["cond"], pair=t["pair"], k=t["k"],
                                     thing=t["thing"], pos=pos, token=r["tokens"][pos], units=uu,
                                     unit_text=[us[u] for u in uu], layer=layer, seed=seed,
                                     states=f"held_search/{args.model}/{t['trial']}.pt", round=1))
    results.folder("held_search")
    write_json(OUT / f"plan_{args.model}.json", plan)
    n_trials = len({e["trial"] for e in plan})
    print(f"{len(plan)} pictures planned in {n_trials} trials ({sum(1 for t in want if not rep[t['trial']]['kept'])} "
          f"grid trials not kept) -> plan_{args.model}.json")


def plans(model):
    out = {}
    for f in sorted(OUT.glob(f"plan_{model}*.json")):   # (*.json: never the .tmp files of write_json)
        for e in json.loads(f.read_text()):
            out.setdefault(e["file"], e)
    return list(out.values())


# ------------------------------------------------------------------------------------------------------------- draw
def step_draw(args):
    import torch
    from vlens import readout
    from vlens.models import load
    todo = [e for e in plans(args.model) if not (OUT / e["file"]).exists()
            and zlib.crc32(e["file"].encode()) % args.of == args.shard]
    # one trial at a time (its states are loaded once), trials in a fixed hashed order so that a partial run covers
    # every sentence alike; a model's draw_first_layer (HunyuanImage: 25) before its other layers
    first = cfg()["models"][args.model].get("draw_first_layer")
    todo.sort(key=lambda e: (e["layer"] != first, zlib.crc32(e["states"].encode()), e["states"], e["seed"], e["pos"],
                             e["layer"]))
    print(len(todo), "pictures to draw", flush=True)
    if not todo:
        return
    m = load(args.model)
    lens = readout.Lens(m, args.model)
    (OUT / args.model).mkdir(parents=True, exist_ok=True)
    states, jobs, t0, n = {}, [], time.time(), 0
    for e in todo:
        if (OUT / e["file"]).exists():   # drawn by another job meanwhile
            continue
        if e["states"] not in states:
            states.clear()   # the list is sorted by trial: one trial's states at a time
            states[e["states"]] = torch.load(paths.DATA / e["states"], map_location="cpu")
        h = states[e["states"]]["hidden"][e["layer"]][e["pos"]].float()
        jobs.append((e["file"].split("/", 1)[1], lens.reading(h, e["layer"]), e["seed"]))
        if len(jobs) >= max(6, 4 * args.batch):
            readout.draw_all(m, jobs, OUT / args.model, args.batch)
            n += len(jobs)
            jobs.clear()
            print(f"[{time.time() - t0:6.0f}s] {n} of {len(todo)}", flush=True)
    readout.draw_all(m, jobs, OUT / args.model, args.batch)
    print(f"[{time.time() - t0:6.0f}s] {len(todo)} pictures -> {OUT / args.model}", flush=True)


# ------------------------------------------------------------------------------------------------------------ judge
def answers(model):
    """{judge id: {judge: chosen}}"""
    out = defaultdict(dict)
    for j in PANEL:
        for f in OUT.glob(f"judge_{j}_{model}_*.json"):
            try:
                rows = json.loads(f.read_text())
            except json.JSONDecodeError:   # a file being written: read again next time
                continue
            for a in rows:
                out[a["jid"]][j] = a["chosen"]
    return out


def all_items(model, c=None):
    c = c or cfg()
    return [(jid, names, e) for e in plans(model) for jid, names in judge_items(e, c, model)]


def step_judge(args):
    from PIL import Image
    from vlens.judges import Judge, subject_options, subject_question
    if args.judge is None:
        raise SystemExit("judge: give --judge (qwen3vl, gemma3 or llavaov2)")
    c, ans = cfg(), answers(args.model)
    todo = [(jid, names, e) for jid, names, e in all_items(args.model, c)
            if args.judge not in ans.get(jid, {}) and (OUT / e["file"]).exists()
            and zlib.crc32(jid.encode()) % args.of == args.shard]
    print(f"{len(todo)} judge items", flush=True)
    if not todo:
        return
    j = Judge(args.judge)
    f = OUT / f"judge_{args.judge}_{args.model}_{args.shard}of{args.of}_{time.strftime('%m%d%H%M%S')}.json"
    out, t0 = [], time.time()
    for jid, names, e in todo:
        try:   # a picture another job is still writing is left for the next judging round
            img = Image.open(OUT / e["file"])
            img.load()
        except OSError:
            print("not readable yet:", e["file"], flush=True)
            continue
        opts = subject_options(names[0], names[1], jid)
        out.append(dict(jid=jid, file=e["file"], names=names, chosen=j.choose(img, opts, subject_question).chosen))
        if len(out) % 50 == 0:
            write_json(f, out)
            print(f"[{time.time() - t0:6.0f}s] {len(out)} of {len(todo)}", flush=True)
    write_json(f, out)
    print(len(out), "->", f, flush=True)


# ------------------------------------------------------------------------------------------------------------ score
class Verdicts:
    """What the judges say: a picture shows X when at least two of the three judges choose X alone."""

    def __init__(self, model, c=None):
        self.c = c or cfg()
        self.model = model
        self.ans = answers(model)
        self.entries = plans(model)
        self.by_cell = defaultdict(dict)   # (trial, pos, layer) -> {seed: entry}
        for e in self.entries:
            self.by_cell[(e["trial"], e["pos"], e["layer"])][e["seed"]] = e
        self.pair_of = {t: p for p, pair in enumerate(self.c["pairs"]) for t in pair}

    def picture(self, e, x):
        """True/False whether the picture of plan entry e shows x, None when not judged by all three yet."""
        if e["cond"] == "think":
            jid = e["file"]
            if x not in self.c["pairs"][e["pair"]]:
                raise ValueError(f"{x} is not asked about in {jid}")
        else:
            jid = f"{e['file']}|p{self.pair_of[x]}"
        v = self.ans.get(jid, {})
        if len(v) < 3 or not (OUT / e["file"]).exists():
            return None
        top = Counter(v.values()).most_common(1)[0]
        return top[1] >= 2 and top[0] == x

    def votes(self, e, x):
        jid = e["file"] if e["cond"] == "think" else f"{e['file']}|p{self.pair_of[x]}"
        return sum(1 for v in self.ans.get(jid, {}).values() if v == x)

    def cell(self, trial, pos, layer, x):
        """The 2-of-3 rule over seeds 0, 1, 2: (decision True/False/None, seeds showing x, seeds judged)."""
        seeds = self.by_cell.get((trial, pos, layer), {})
        got = [self.picture(seeds[s], x) for s in (0, 1, 2) if s in seeds]
        yes, no = sum(g is True for g in got), sum(g is False for g in got)
        return (True if yes >= 2 else False if no >= 2 else None), yes, yes + no


def step_adapt(args):
    """Seed 2 wherever seeds 0 and 1 are both judged and disagree on any verdict that is scored from that cell: in a think
    trial whether it shows its thing and whether it shows the partner (the partner's swap control), in a no-instruction
    trial whether it shows any thing of the grid. Exactly the 2-of-3 rule; cells not yet judged are reported."""
    c = cfg()
    V = Verdicts(args.model, c)
    grid = c["models"][args.model]
    if "adaptive_seed" not in grid:
        raise SystemExit(f"{args.model} draws all seeds up front; adapt is for HunyuanImage")
    things = [t for p in grid["pairs"] for t in c["pairs"][p]]
    add, waiting = [], 0
    for (trial, pos, layer), seeds in sorted(V.by_cell.items()):
        e0 = seeds.get(0)
        if e0 is None or e0["round"] != 1 or 2 in seeds:
            continue
        xs = c["pairs"][e0["pair"]] if e0["cond"] == "think" else things
        v = [[V.picture(seeds[s], x) for x in xs] for s in (0, 1)]
        if any(a is None for row in v for a in row):
            waiting += 1
            continue
        if v[0] != v[1]:
            add.append(dict(e0, file=picture(args.model, trial, pos, layer, grid["adaptive_seed"]),
                            seed=grid["adaptive_seed"], round=2))
    f = OUT / f"plan_{args.model}_r2.json"
    old = json.loads(f.read_text()) if f.exists() else []
    have = {e["file"] for e in old}
    new = old + [e for e in add if e["file"] not in have]
    write_json(f, new)
    print(json.dumps(dict(seed2=len(new), added=len(new) - len(old), waiting=waiting)))


def sign_test(n10, n01):
    """Two-sided exact binomial test on the discordant pairs."""
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def score_model(model, c):
    V = Verdicts(model, c)
    rep, sents = replies(model), sentences(c)
    grid = c["models"][model]
    layers = grid["layers"]
    rows, undecided_cells = [], 0
    for sid in grid["sentences"]:
        us = units(sents[sid]["text"])
        none = rep.get(trial_id(sid))
        for p in grid["pairs"]:
            for k, thing in enumerate(c["pairs"][p]):
                partner = c["pairs"][p][1 - k]
                think, swap = rep.get(trial_id(sid, thing)), rep.get(trial_id(sid, partner))
                row = dict(sid=sid, extra=bool(sents[sid].get("extra")), added=bool(sents[sid].get("added")),
                           thing=thing, partner=partner,
                           think_kept=bool(think and think["kept"]), swap_kept=bool(swap and swap["kept"]),
                           none_kept=bool(none and none["kept"]))

                def scan(r, cond_kept):
                    """Every (unit, layer) cell of trial r for 'shows thing': list of dicts."""
                    if not cond_kept:
                        return []
                    out = []
                    for u, pos in enumerate(r["units"]):
                        if pos is None:
                            continue
                        for layer in layers:
                            d, yes, n = V.cell(r["trial"], pos, layer, thing)
                            out.append(dict(unit=u, unit_text=us[u], pos=pos, token=r["tokens"][pos], layer=layer,
                                            decision=d, seeds_showing=yes, seeds_judged=n))
                    return out

                cells = {"think": scan(think, row["think_kept"]), "swap": scan(swap, row["swap_kept"]),
                         "none": scan(none, row["none_kept"])}
                for cond, cl in cells.items():
                    hit = any(x["decision"] for x in cl)
                    und = sum(x["decision"] is None for x in cl)
                    undecided_cells += und
                    row[cond] = (None if not cl else True if hit else (None if und else False))
                    row[f"{cond}_undecided_cells"] = und
                    row[f"{cond}_cells"] = len(cl)
                # every hit cell of the think trial, with its matched controls (same unit and layer)
                matched = {cond: {(x["unit"], x["layer"]): x for x in cells[cond]} for cond in ("swap", "none")}
                hits = []
                for x in cells["think"]:
                    if not x["decision"]:
                        continue
                    sw, no = matched["swap"].get((x["unit"], x["layer"])), matched["none"].get((x["unit"], x["layer"]))
                    seeds = V.by_cell[(think["trial"], x["pos"], x["layer"])]
                    hits.append(dict(x, votes=sum(V.votes(seeds[s], thing) for s in seeds),
                                     swap=None if sw is None else [sw["decision"], sw["seeds_showing"], sw["seeds_judged"]],
                                     none=None if no is None else [no["decision"], no["seeds_showing"], no["seeds_judged"]],
                                     clean=bool(sw and no and sw["decision"] is False and no["decision"] is False),
                                     think_trial=think["trial"], swap_trial=swap["trial"] if sw else None,
                                     none_trial=none["trial"] if no else None, kind=kind(x["unit_text"])))
                row["hit_cells"] = hits
                row["clean"] = any(h["clean"] for h in hits)
                rows.append(row)
    return rows, undecided_cells


def summarize(rows):
    main = [r for r in rows if not r["extra"]]
    kept = [r for r in main if r["think_kept"]]
    s = dict(instructions=len(main), kept=len(kept), success=sum(r["think"] is True for r in kept),
             clean_success=sum(r["clean"] for r in kept), undecided=sum(r["think"] is None for r in kept))
    sw = [r for r in kept if r["swap_kept"]]
    no = [r for r in main if r["none_kept"]]
    s.update(swap_controls=len(sw), swap_false_positive=sum(r["swap"] is True for r in sw),
             none_controls=len(no), none_false_positive=sum(r["none"] is True for r in no))
    n10 = sum(r["think"] is True and r["swap"] is False for r in sw)
    n01 = sum(r["think"] is False and r["swap"] is True for r in sw)
    s.update(paired_think_only=n10, paired_swap_only=n01, sign_test_p=round(sign_test(n10, n01), 5))
    hit_cells = [h for r in kept for h in r["hit_cells"]]
    s["hit_cells"] = len(hit_cells)
    s["clean_hit_cells"] = sum(h["clean"] for h in hit_cells)
    s["hit_cells_by_layer"] = dict(Counter(h["layer"] for h in hit_cells))
    s["hit_cells_by_kind"] = dict(Counter(h["kind"] for h in hit_cells))
    s["success_by_thing"] = {t: [sum(r["think"] is True for r in kept if r["thing"] == t),
                                 sum(1 for r in kept if r["thing"] == t)] for t in sorted({r["thing"] for r in main})}
    return s


def best_cells(rows, extra=False):
    """Every success's best hit cell, best first: clean controls, then seeds showing the thing, then judge votes."""
    out = []
    for r in rows:
        if r["extra"] != extra or not r["hit_cells"]:
            continue
        h = max(r["hit_cells"], key=lambda h: (h["clean"], h["seeds_showing"], h["votes"],
                                               -(h["swap"][1] if h["swap"] else 9) - (h["none"][1] if h["none"] else 9)))
        out.append(dict(h, sid=r["sid"], thing=r["thing"], partner=r["partner"], n_hit_cells=len(r["hit_cells"])))
    return sorted(out, key=lambda h: (-h["clean"], -h["seeds_showing"], -h["votes"], h["sid"], h["thing"]))


def step_score(args):
    c = cfg()
    sents = sentences(c)
    for model in args.model.split(","):
        rows, und = score_model(model, c)
        summ = summarize(rows)
        best = best_cells(rows)
        rep = replies(model)
        copy = dict(all_prompts=len(rep), kept=sum(r["kept"] for r in rep.values()),
                    grid_think=[sum(r["kept"] for r in rep.values() if r["cond"] == "think" and r["grid"]),
                                sum(1 for r in rep.values() if r["cond"] == "think" and r["grid"])])
        main = [r for r in rows if not r["extra"]]
        subsets = {"s01-s10": [r for r in main if not r["added"]], "s11-s14": [r for r in main if r["added"]]}
        out = dict(model=model, summary=summ, subsets={k: summarize(v) for k, v in subsets.items() if v},
                   extra=summarize([dict(r, extra=False) for r in rows if r["extra"]])
                   if any(r["extra"] for r in rows) else None,
                   undecided_cells=und, copy=copy, best=best, best_extra=best_cells(rows, extra=True), rows=rows)
        (OUT / f"scores_{model}.json").write_text(json.dumps(out, indent=1))
        print(f"== {model}: copies kept {copy}")
        print(json.dumps({k: v for k, v in summ.items() if not isinstance(v, dict)}))
        for k in ("hit_cells_by_layer", "hit_cells_by_kind", "success_by_thing"):
            print(" ", k, summ[k])
        print("  undecided cells:", und)
        for h in best[:40]:
            print(f"  {h['sid']} {sents[h['sid']]['text']!r} think {h['thing']!r} token {h['token']!r} (unit "
                  f"{h['unit_text']!r}) L{h['layer']}: {h['seeds_showing']}/3 seeds, votes {h['votes']}, swap {h['swap']}, "
                  f"none {h['none']}, clean {h['clean']}, hit cells {h['n_hit_cells']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["reply", "plan", "draw", "judge", "adapt", "score"])
    ap.add_argument("--model", default="show_o2")
    ap.add_argument("--batch", type=int, default=1, help="draw: pictures per batch")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--grid-only", action="store_true", help="reply: only the trials of the model's grid")
    ap.add_argument("--judge", choices=PANEL)
    args = ap.parse_args()
    dict(reply=step_reply, plan=step_plan, draw=step_draw, judge=step_judge, adapt=step_adapt,
         score=step_score)[args.step](args)


if __name__ == "__main__":
    main()
