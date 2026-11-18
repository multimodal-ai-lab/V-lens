"""The described object at every token after the question, through the first answer token (Sec. 4.3, Fig. 4 and
Supp. D, Table S11).

Already drawn and judged, inferred condition, the questions scripts/depth_fixed_positions.py counts:
- scripts/depth_fixed_positions.py: the describing word's last token ("desc"), the question mark ("end"; HunyuanImage's
  comes from scripts/depth_all_tokens.py) and the token right before the answer ("preanswer");
- scripts/depth_all_tokens.py: every other question token ("tokNN").
Drawn here: every token the model reads after the question mark and before the token right before the answer when it
answers (the " Answer with one word." it is asked with, then its chat format: end of the user turn, assistant marker,
empty thinking block, ...; position "postNNN", NNN its index in that reading), and the first token of its answer
("answer1"). All from one teacher-forced reading: the question and the instruction in the chat format the answer was given
in, followed by the reply recorded by scripts/depth_fixed_positions.py (HunyuanImage: with its one-word system prompt).
Same layers, request, insertion layer, V-lens (vlens.readout.Lens) and seeds as scripts/depth_all_tokens.py: seeds 3 and
4, then seed 5 only where their verdicts differ (two equal verdicts decide two of three).

    plan (GPU, once per model): the readings, positions and token ids -> <model>/plan.json, plus checks (the question
        mark's state against the plain reading of the other two scripts; HunyuanImage: the teacher-forced pass against
        the greedy regeneration). --smoke L1,L2 also draws seed 3 of the first two questions' answer token, first chat
        token and last instruction token at those layers into smoke/<model>/ (not judged).
    draw (GPU): --shard/--of over the (question, position) units of plan.json in a fixed order (answer and chat tokens
        before the instruction's); --seeds 3,4 --tag a; stage 2: --need <file> --seeds 5 --tag b. Writes
        done_<tag><shard>of<of> when every picture of the shard exists.
    need (CPU): the cells whose seeds 3 and 4 disagree -> <model>/need.json (exit 3 while a picture lacks a judge).
    unjudged (CPU): pictures without each judge's answer.
    score (CPU): the best later token per model and layer (Fig. 4; the curve "best_later"), the earlier question-only
        curve, the answer token, and which kinds of positions show the object (Table S11) -> scores.json,
        answer_token_pictures.json.

    python scripts/depth_through_answer.py plan --model show_o2
    python scripts/depth_through_answer.py draw --model show_o2 --shard 0 --of 8 --seeds 3,4 --tag a
    python scripts/run_judge.py --judge qwen3vl --set cells:alltokens_answer --subject --model show_o2   (and gemma3, llavaov2)
    python scripts/depth_through_answer.py need --model show_o2
    python scripts/depth_through_answer.py draw --model show_o2 --need $VLENS_OUT/alltokens_answer/show_o2/need.json \\
        --seeds 5 --tag b
    (judge again)
    python scripts/depth_through_answer.py score
    python scripts/depth_token_report.py
"""

import argparse
import functools
import importlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from vlens import readout, results

OUT = results.RESULTS / "alltokens_answer"
CONFIRM = results.RESULTS / "bridge_confirm"
ALLTOK = results.RESULTS / "bridge_alltokens"
JUDGES = results.RESULTS / "judges"
PANEL = ("qwen3vl", "gemma3", "llavaov2")
FOLDER = "alltokens_answer"                                   # the judge set of this run: cells:alltokens_answer
FOLDERS = ("bridge_confirm", "bridge_alltokens", FOLDER)     # every judge set the merged cells come from
MODELS = ("bagel", "show_o2", "sensenova", "hunyuan")
RANK = {"answer": 0, "chat": 1, "instruction": 2}            # drawing order of the position groups


all_tokens = results.load_script("depth_all_tokens")
fixed = all_tokens.fixed


# --------------------------------------------------------------------------------------------- positions (CPU, tested)
def char_starts(strs):
    starts, full = [], ""
    for s in strs:
        starts.append(len(full))
        full += s
    return starts, full


def question_spans(strs, text, asked):
    """In a reading's token strings: the token holding the question's last character (its question mark), and the
    tokens of the instruction that follows it inside the same message (" Answer with one word.")."""
    starts, full = char_starts(strs)
    a = full.rfind(asked)
    if a < 0:
        raise ValueError(f"the asked text is not in the reading: {asked!r}")
    last = a + len(text) - 1
    q_end = [p for p, s0 in enumerate(starts) if s0 <= last < s0 + len(strs[p])][0]
    lo, hi = a + len(text), a + len(asked)
    instr = {p for p, s0 in enumerate(starts) if strs[p] and s0 >= lo and s0 + len(strs[p]) <= hi}
    return q_end, instr


def answer_spot(strs, content, reply, answers):
    """The rule of scripts/depth_fixed_positions.py (step_draw): among the reply's content tokens, the first that reaches the first right answer in the
    reply; returns (the token right before it, the token itself)."""
    hit = readout.answer_hit(reply, answers)
    if hit is None:
        raise ValueError(f"no right answer in {reply!r}")
    rc = [q for q in range(len(strs)) if content[q]]
    starts, full = [], ""
    for q in rc:
        starts.append(len(full))
        full += strs[q]
    off = len(full) - len(full.lstrip())
    k = [n for n, s0 in enumerate(starts) if s0 + len(strs[rc[n]]) > hit[0] + off][0]
    return rc[k] - 1, rc[k]


def new_positions(strs, q_end, instr, pre, ans):
    """Every position after the question mark and before the token right before the answer, and the answer's first
    token, with its group: instruction (" Answer with one word."), chat (the chat format between the user's message and
    the answer), answer."""
    assert q_end < pre < ans, (q_end, pre, ans)
    out = [dict(p=p, group="instruction" if p in instr else "chat", token=strs[p]) for p in range(q_end + 1, pre)]
    return out + [dict(p=ans, group="answer", token=strs[ans])]


def position_name(group, p):
    return "answer1" if group == "answer" else f"post{p:03d}"


def units_of(plan):
    """(question entry, position entry) in the drawing order: answer and chat tokens first, then the instruction's."""
    u = [(q, pos) for q in plan["questions"] for pos in q["positions"]]
    return sorted(u, key=lambda x: (RANK[x[1]["group"]], x[0]["item"], x[1]["p"]))


def shard_slice(n, shard, of):
    """Contiguous shards, so shard 0 holds the first units of the drawing order."""
    return slice(shard * n // of, (shard + 1) * n // of)


# --------------------------------------------------------------------------------------------- readings (GPU)
def recorded(model):
    """The fixed-position run's inferred replies and preanswer tokens, by item, and the counted questions."""
    ans, pre = {}, {}
    for f in (CONFIRM / model).glob("answers_*.json"):
        for a in json.loads(f.read_text()):
            if a["condition"] == "inferred":
                ans[a["item"]] = a
    for f in (CONFIRM / model).glob("records_*.json"):
        for r in json.loads(f.read_text()):
            if r["kind"] == "reads" and r.get("condition") == "inferred" and r["position"] == "preanswer":
                pre[r["item"]] = r["token"]
    return all_tokens.counted(model), ans, pre


def hy_forced(m, asked, gen_ids):
    """HunyuanImage, teacher-forced: the one-word prompt (gen_text, the one-word system prompt) with `gen_ids` appended, read
    in one prefill pass (the construction of the adapter's reply_loss, in its capture mode); stops after that pass."""
    import torch
    hy = importlib.import_module("vlens.models.hunyuan")
    inputs = m.net.prepare_model_inputs(prompt=asked, mode="gen_text", bot_task="auto", max_new_tokens=len(gen_ids) + 1,
                                        system_prompt=fixed.HUNYUAN_SYSTEM)
    n = inputs["input_ids"].shape[1]
    if gen_ids:
        ids = torch.tensor([list(gen_ids)], device=inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], 1)
        inputs["position_ids"] = torch.arange(n + len(gen_ids), device=ids.device)[None]
    inputs["max_new_tokens"] = 1
    orig = m.net.forward

    @functools.wraps(orig)   # generate() checks its arguments against the forward's signature
    def forward(*a, **kw):
        orig(*a, **kw)
        raise hy._Stop
    m._arm("capture")
    m.net.forward = forward
    try:
        with torch.no_grad():
            m.net.generate(**inputs, decode_text=True, do_sample=False)
    except hy._Stop:
        pass
    finally:
        del m.net.forward
        m._armed = False
    reading = m._reading(asked, m._ids[m._rows[0]].tolist())
    return reading, n


class Seq:
    """One answer-time sequence: token ids and strings, and the hidden state entering each layer at each position."""

    def __init__(self, ids, strs, parts):
        self.ids, self.strs, self.parts = ids, strs, parts   # parts: [(offset, reading)]

    def hidden(self, layer, p):
        for off, rd in reversed(self.parts):
            if p >= off:
                return rd.hidden[layer][p - off]
        raise IndexError(p)


def answer_reading(m, model, asked, reply, gen_ids=None):
    """The reading of the question (with the instruction) in the chat format the reply was given in, and the reply
    teacher-forced after it. Returns (Seq, content mask over the sequence marking the reply's tokens)."""
    if model == "bagel":   # read_reply's two messages, kept both
        first = m.read(asked)
        rep = m.read(reply, previous=first.cache)
        n = len(first.token_ids)
        content = [False] * n + [bool(c) for c in rep.content]
        return Seq(first.token_ids + rep.token_ids, first.token_strs + rep.token_strs, [(0, first), (n, rep)]), content
    if model in ("show_o2", "sensenova"):
        rep = m.read_reply(asked, reply)
        return Seq(rep.token_ids, rep.token_strs, [(0, rep)]), [bool(c) for c in rep.content]
    if model == "hunyuan":
        rd, n = hy_forced(m, asked, gen_ids)
        content = [False] * n + [bool(s.strip()) and s.strip() not in ("<answer>", "</answer>", "<|endoftext|>")
                                 for s in rd.token_strs[n:]]
        return Seq(rd.token_ids, rd.token_strs, [(0, rd)]), content
    raise ValueError(model)


def plain_reading(m, model, text):
    return m.read(text, system=fixed.HUNYUAN_SYSTEM) if model == "hunyuan" else m.read(text)


def cos(a, b):
    a, b = a.float().flatten().cpu(), b.float().flatten().cpu()
    return float((a @ b) / (a.norm() * b.norm()))


def step_plan(args):
    import torch
    from vlens.models import load
    qs = dict(fixed.items()[0])
    use, ans, pre_rec = recorded(args.model)
    m = load(args.model)
    hy = args.model == "hunyuan"
    L = list(range(0, m.n_layers, 3)) + [m.n_layers - 1] if hy else readout.layers(m.n_layers)
    plan = dict(model=args.model, layers=L, questions=[], checks={})
    for i in use:
        it, reply = qs[i], ans[i]["reply"]
        text = it["text"]
        asked = text + readout.ONE_WORD
        gen_ids, forced = None, False
        if hy:   # the greedy regeneration the fixed-position run used (read_reply); its reply must be the recorded one
            again, follow = m._follow(asked, fixed.HUNYUAN_SYSTEM, 8)
            same = again == reply.strip()
            fcontent = [bool(c) for c in follow.content]
            fpre, fans = answer_spot(follow.token_strs, fcontent, again, it["answers"]) if same else (None, None)
            prompt = m.net.prepare_model_inputs(prompt=asked, mode="gen_text", bot_task="auto", max_new_tokens=1,
                                                system_prompt=fixed.HUNYUAN_SYSTEM)["input_ids"].shape[1]
            if same:
                gen_ids = follow.token_ids[prompt:fans + 1]
            else:   # rebuild the reply's first tokens from the recorded reply and the recorded preanswer token
                forced = True
                hit = readout.answer_hit(reply, it["answers"])
                tk = m.tokenizer
                gen_ids = []
                if pre_rec[i] != follow.token_strs[prompt - 1]:
                    gen_ids += tk.encode(pre_rec[i], add_special_tokens=False)
                gen_ids += tk.encode(reply[hit[0]:], add_special_tokens=False)[:1]
                print(f"  {it['id']}: regenerated {again!r} differs from recorded {reply!r}; teacher-forced ids "
                      f"{gen_ids} {[tk.decode([g]) for g in gen_ids]}", flush=True)
        seq, content = answer_reading(m, args.model, asked, reply, gen_ids)
        q_end, instr = question_spans(seq.strs, text, asked)
        pre, a1 = answer_spot(seq.strs, content, reply, it["answers"])
        assert seq.strs[q_end].strip().endswith("?"), seq.strs[q_end]
        if seq.strs[pre] != pre_rec[i]:
            raise RuntimeError(f"{it['id']}: preanswer token {seq.strs[pre]!r} differs from the recorded {pre_rec[i]!r}")
        positions = new_positions(seq.strs, q_end, instr, pre, a1)
        # checks: the question mark's state against the plain reading of the other two scripts (the same tokens up
        # to it in BAGEL, SenseNova-U1 and HunyuanImage; Show-o2's plain reading has no chat header); HunyuanImage:
        # the teacher-forced states against the greedy regeneration's, at every position up to the answer token
        plain = plain_reading(m, args.model, text)
        pc = [q for q in range(len(plain.token_ids)) if plain.content[q]]
        chk = dict(q_mark_cos={l: round(cos(seq.hidden(l, q_end), plain.hidden[l][pc[-1]]), 5) for l in L})
        if hy and not forced:
            chk["forced_vs_follow_min_cos"] = round(min(cos(seq.hidden(l, p), follow.hidden[l][p])
                                                        for l in L for p in range(a1 + 1)), 5)
            assert seq.ids[:a1 + 1] == follow.token_ids[:a1 + 1]
            assert (fpre, fans) == (pre, a1)
        plan["checks"][it["id"]] = chk
        plan["questions"].append(dict(item=i, id=it["id"], text=text, asked=asked, reply=reply, forced=forced,
                                      ids=seq.ids[:a1 + 1], strs=seq.strs[:a1 + 1], q_end=q_end, pre=pre, ans=a1,
                                      gen_ids=gen_ids, positions=positions))
        print(f"{it['id']:16s} {reply!r:12s} ?={q_end} pre={pre} {seq.strs[pre]!r} ans={a1} {seq.strs[a1]!r}  new "
              f"{[(x['token'], x['group'][0]) for x in positions]}  ?-cos {min(chk['q_mark_cos'].values()):.4f}"
              + (f"  forced/follow {chk.get('forced_vs_follow_min_cos')}" if hy else ""), flush=True)
        torch.cuda.empty_cache()
    n_units = sum(len(q["positions"]) for q in plan["questions"])
    plan["units"], plan["pictures_stage1"] = n_units, n_units * len(L) * 2
    (OUT / args.model).mkdir(parents=True, exist_ok=True)
    (OUT / args.model / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"{args.model}: {len(plan['questions'])} questions, {n_units} positions, {len(L)} layers, "
          f"{plan['pictures_stage1']} pictures for seeds 3 and 4", flush=True)
    if args.smoke:
        smoke(m, args, plan)


def smoke(m, args, plan):
    """Seed 3 of the first two questions' answer token, first chat token and last instruction token, at --smoke layers."""
    lens = readout.Lens(m, args.model)
    out = OUT / "smoke" / args.model
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.smoke.split(",")]
    jobs, recs = [], []
    for q in plan["questions"][:2]:
        seq, _ = answer_reading(m, args.model, q["asked"], q["reply"], q["gen_ids"])
        assert seq.ids[:q["ans"] + 1] == q["ids"]
        pick = [x for x in q["positions"] if x["group"] == "answer"]
        pick += [x for x in q["positions"] if x["group"] == "chat"][:1]
        pick += [x for x in q["positions"] if x["group"] == "instruction"][-1:]
        for x in pick:
            for l in layers:
                f = f"{q['id']}_{position_name(x['group'], x['p'])}_L{l:02d}_s3.png"
                jobs.append((f, lens.reading(seq.hidden(l, x["p"]), l), 3))
                recs.append(dict(file=f, token=x["token"], group=x["group"], layer=l))
    t0 = time.time()
    readout.draw_all(m, jobs, out, 1 if args.model == "hunyuan" else 4)
    (out / "records_smoke.json").write_text(json.dumps(recs, indent=1))
    import torch
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
    print(f"smoke: {len(jobs)} pictures in {time.time() - t0:.0f} s, peak memory {mem:.1f} GiB -> {out}", flush=True)


def step_draw(args):
    import torch
    from vlens.models import load
    plan = json.loads((OUT / args.model / "plan.json").read_text())
    qs = dict(fixed.items()[0])
    need = None
    if args.need:
        need = {tuple(c) for c in json.loads(Path(args.need).read_text())["cells"]}
    seeds = [int(s) for s in args.seeds.split(",")]
    units = units_of(plan)
    mine = units[shard_slice(len(units), args.shard, args.of)]
    if need is not None:
        mine = [(q, x) for q, x in mine if any((q["item"], position_name(x["group"], x["p"]), l) in need
                                                for l in plan["layers"])]
    tag = f"{args.tag}{args.shard}of{args.of}"
    out = OUT / args.model
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(mine)} of {len(units)} units in shard {args.shard}/{args.of}", flush=True)
    if not mine:
        (out / f"records_{tag}.json").write_text("[]")
        (out / f"done_{tag}").write_text("0 pictures\n")
        return
    m = load(args.model)
    lens = readout.Lens(m, args.model)
    records, jobs, t0 = [], [], time.time()
    reading_of = {}

    def flush():
        readout.draw_all(m, jobs, out, args.batch)
        jobs.clear()
        torch.cuda.empty_cache()
        (out / f"records_{tag}.json").write_text(json.dumps(records, indent=1))

    for q, x in mine:
        if q["item"] not in reading_of:   # one question's reading at a time
            reading_of.clear()
            seq, _ = answer_reading(m, args.model, q["asked"], q["reply"], q["gen_ids"])
            if seq.ids[:q["ans"] + 1] != q["ids"]:
                raise RuntimeError(f"{q['id']}: the reading's tokens differ from plan.json")
            reading_of[q["item"]] = seq
        seq = reading_of[q["item"]]
        pos = position_name(x["group"], x["p"])
        for l in plan["layers"]:
            if need is not None and (q["item"], pos, l) not in need:
                continue
            r = lens.reading(seq.hidden(l, x["p"]), l)
            for seed in seeds:
                f = f"{q['id']}_inferred_{pos}_L{l:02d}_s{seed}.png"
                jobs.append((f, r, seed))
                it = qs[q["item"]]
                records.append(dict(file=f"{args.model}/{f}", model=args.model, items="bridges", kind="reads",
                                    item=q["item"], family=it["family"], condition="inferred", position=pos, layer=l,
                                    way="lens", seed=seed, token=x["token"], group=x["group"], index=x["p"],
                                    q_end=q["q_end"], pre=q["pre"], ans=q["ans"], forced=q["forced"],
                                    truth=it["truth"], alt=it["alt"]))
        flush()
        print(f"[{time.time() - t0:6.0f}s] {q['id']} {pos} {x['token']!r} ({x['group']})", flush=True)
    flush()
    missing = [r["file"] for r in records if not (out / r["file"].split("/", 1)[1]).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} pictures missing after drawing")
    (out / f"done_{tag}").write_text(f"{len(records)} pictures\n")
    print(len(records), "pictures ->", out)


# --------------------------------------------------------------------------------------------- judging and scoring (CPU)
def votes(folders=FOLDERS):
    v = defaultdict(dict)
    for folder in folders:
        for j in PANEL:
            for f in JUDGES.glob(f"answers_{j}_cells_{folder}_subject_*.json"):
                for a in json.loads(f.read_text()):
                    v[a["file"]][j] = a["chosen"]
    return v


def own_records(model, stage=None):
    recs = {}
    for f in sorted((OUT / model).glob("records_*.json")):
        if stage and not f.name.startswith(f"records_{stage}"):
            continue
        for r in json.loads(f.read_text()):
            recs[r["file"]] = r
    return recs


def step_unjudged(args):
    """Pictures without each judge's answer, as "qwen3vl gemma3 llavaov2" counts on one line."""
    v = votes((FOLDER,))
    recs = own_records(args.model)
    print(" ".join(str(sum(1 for fn in recs if j not in v.get(fn, {}))) for j in PANEL))


def step_need(args):
    v = votes((FOLDER,))
    cells, unjudged = defaultdict(dict), 0
    for fn, r in own_records(args.model, "a").items():
        verdict = all_tokens.shows(v.get(fn, {}), r["truth"])
        unjudged += verdict is None
        cells[(r["item"], r["position"], r["layer"])][r["seed"]] = verdict
    need = sorted(k for k, s in cells.items() if s.get(3) is not None and s.get(4) is not None and s[3] != s[4])
    out = Path(args.out or OUT / args.model / "need.json")
    if unjudged:
        print(f"{unjudged} stage-1 pictures lack a judge; no need file written")
        raise SystemExit(3)
    out.write_text(json.dumps(dict(model=args.model, cells=[list(k) for k in need]), indent=1))
    print(f"{len(need)} of {len(cells)} cells need seed 5 -> {out}")


def kind_of(r, last_desc):
    """The kind of a read position after the describing word (None for the word itself and anything before it)."""
    pos = r["position"]
    if pos == "desc":
        return None
    if pos == "end":
        return "question mark"
    if pos.startswith("tok"):
        p = int(pos[3:])
        if p <= last_desc:
            return None
        return "question mark" if r["token"].strip() == "?" else "question word"
    if pos == "preanswer":
        return "pre-answer"
    if pos == "answer1":
        return "answer token"
    if pos.startswith("post"):
        return r["group"]   # instruction or chat
    return None


def cell_table(model, use, v):
    """Every inferred read of the counted questions from the three runs: cells (item, position, layer) -> {seed:
    verdict}, and one record per cell."""
    recs = {}
    for folder in (CONFIRM, ALLTOK, OUT):
        for f in (folder / model).glob("records_*.json"):
            for r in json.loads(f.read_text()):
                if r["kind"] == "reads" and r.get("condition") == "inferred" and r["item"] in use:
                    recs[r["file"]] = r
    cells, rec_of = defaultdict(dict), {}
    for fn, r in recs.items():
        k = (r["item"], r["position"], r["layer"])
        cells[k][r["seed"]] = all_tokens.shows(v.get(fn, {}), r["truth"])
        rec_of[k] = r
    return cells, rec_of, recs


def decide(seeds):
    vs = list(seeds.values())
    if vs.count(True) >= 2:
        return True
    if vs.count(False) >= 2:
        return False
    return None


def last_desc_token(cells):
    """The rule of scripts/depth_all_tokens.py: the identifying word's last token is the gap the excluded word leaves among the drawn question
    tokens (or right after them, when the word stands just before the question mark)."""
    toks = defaultdict(set)
    for (i, pos, l) in cells:
        if pos.startswith("tok"):
            toks[i].add(int(pos[3:]))
    return {i: max([p for p in range(min(ix), max(ix)) if p not in ix] or [max(ix) + 1]) for i, ix in toks.items()}


CURVES = {   # name -> the kinds of positions it takes the best of
    "best_later_old": {"question word", "question mark"},
    "best_later": {"question word", "question mark", "instruction", "chat", "pre-answer", "answer token"},
    "best_later_no_instruction": {"question word", "question mark", "chat", "pre-answer", "answer token"},
    "best_later_to_preanswer": {"question word", "question mark", "instruction", "chat", "pre-answer"},
    "best_question": {"question word", "question mark"},
    "best_instruction": {"instruction"},
    "best_chat": {"chat", "pre-answer"},
    "answer_token": {"answer token"},
    "preanswer": {"pre-answer"},
}


def score_model(model, v, qs):
    use = set(all_tokens.counted(model))
    cells, rec_of, recs = cell_table(model, use, v)
    ok = {k: decide(s) for k, s in cells.items()}
    pending = sorted(k for k, d in ok.items() if d is None)
    last = last_desc_token(cells)
    kind = {k: kind_of(rec_of[k], last[k[0]]) for k in cells}
    layers = sorted({k[2] for k in cells if k[1] == "desc"})
    res = dict(counted=len(use), pending_cells=len(pending), layers=layers, desc={}, curves={c: {} for c in CURVES},
               wins={}, tokens_showing={}, sole_new={})
    for l in layers:
        res["desc"][l] = sum(bool(ok.get((i, "desc", l))) for i in use)
        hits = defaultdict(set)   # item -> kinds showing the object at this layer
        toks = defaultdict(list)
        for k, d in ok.items():
            if k[2] == l and d and kind[k]:
                hits[k[0]].add(kind[k])
                toks[k[0]].append(f"{kind[k]}:{rec_of[k]['token']!r}")
        for c, kinds in CURVES.items():
            res["curves"][c][l] = sum(bool(hits[i] & kinds) for i in use)
        res["wins"][l] = dict(Counter(kd for i in use for kd in hits[i]))
        res["tokens_showing"][l] = {qs[i]["id"]: sorted(toks[i]) for i in use if toks[i]}
        # questions the old curve misses and the new one finds
        res["sole_new"][l] = sorted(qs[i]["id"] for i in use if hits[i] & CURVES["best_later"]
                                    and not hits[i] & CURVES["best_later_old"])
    res["best_later"] = res["curves"]["best_later"]            # the curve of Fig. 4
    res["best_later_old"] = res["curves"]["best_later_old"]
    # the answer token: what the judges choose for its pictures (majority of the three, or none), by family
    from vlens.judges import SUBJECT_BOTH, SUBJECT_NEITHER
    ch = defaultdict(Counter)
    opens = []
    for fn, r in recs.items():
        if r["position"] != "answer1":
            continue
        vv = v.get(fn, {})
        if len(vv) < 3:
            continue
        top, n = Counter(vv.values()).most_common(1)[0]
        label = (dict([(r["truth"], "object"), (r["alt"], "alternative"), (SUBJECT_BOTH, "both"),
                       (SUBJECT_NEITHER, "neither")]).get(top, top) if n >= 2 else "no majority")
        ch[r["family"]][label] += 1
        opens.append(dict(file=fn, layer=r["layer"], token=r["token"], truth=r["truth"], alt=r["alt"], verdict=label))
    res["answer_token_choices"] = {f: dict(c) for f, c in ch.items()}
    return res, opens, pending


def open_answers():
    """The judges' free descriptions of each picture of this run (judge -> file -> text)."""
    o = defaultdict(dict)
    for j in PANEL:
        for f in JUDGES.glob(f"answers_{j}_cells_{FOLDER}_subject_*.json"):
            for a in json.loads(f.read_text()):
                o[a["file"]][j] = a.get("open")
    return o


def step_score(args):
    qs = dict(fixed.items()[0])
    v = votes()
    summary, opens_all = {}, []
    old = json.loads((ALLTOK / "scores.json").read_text()) if (ALLTOK / "scores.json").exists() else {}
    for model in [x for x in MODELS if (OUT / x / "plan.json").exists() or args.allow_missing]:
        res, opens, pending = score_model(model, v, qs)
        # check: the question-only curve equals scripts/depth_all_tokens.py's best later token wherever it decided it
        diff = {l: (res["best_later_old"][l], old[model]["best_later"].get(str(l))) for l in res["layers"]
                if old.get(model) and res["best_later_old"][l] != old[model]["best_later"].get(str(l))}
        res["check_old_curve_differs_from_all_tokens"] = diff
        summary[model] = res
        opens_all += opens
        print(f"{model}: counted {res['counted']}, cells waiting {len(pending)}"
              + (f", question-only curve differs from depth_all_tokens at {diff}" if diff
                 else ", question-only curve = depth_all_tokens'"))
        print("  layer  desc  old  new  new-no-instr  to-pre  pre  answer  chat  instr  question")
        for l in res["layers"]:
            c = res["curves"]
            print(f"  L{l:2d}   {res['desc'][l]:4d} {c['best_later_old'][l]:4d} {c['best_later'][l]:4d} "
                  f"{c['best_later_no_instruction'][l]:13d} {c['best_later_to_preanswer'][l]:7d} {c['preanswer'][l]:4d} "
                  f"{c['answer_token'][l]:7d} {c['best_chat'][l]:5d} {c['best_instruction'][l]:6d} "
                  f"{c['best_question'][l]:9d}   {res['wins'][l]}")
        print("  answer token, judges' choice:", res["answer_token_choices"])
    oa = open_answers()
    for o in opens_all:
        o["open"] = oa.get(o["file"], {})
    (OUT / "scores.json").write_text(json.dumps(summary, indent=1))
    (OUT / "answer_token_pictures.json").write_text(json.dumps(sorted(opens_all, key=lambda o: (o["file"])), indent=1))
    print("->", OUT / "scores.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["plan", "draw", "need", "unjudged", "score"])
    ap.add_argument("--model", default="show_o2", choices=MODELS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seeds", default="3,4")
    ap.add_argument("--tag", default="a", help="records/done file prefix: a for seeds 3 and 4, b for seed 5")
    ap.add_argument("--need", default=None, help="draw: only the cells of this need file (stage 2)")
    ap.add_argument("--smoke", default=None, help="plan: also draw seed 3 of a few positions at these layers")
    ap.add_argument("--out", default=None, help="need: the need file (default <model>/need.json)")
    ap.add_argument("--allow-missing", action="store_true", help="score: also models without a plan (old curves only)")
    args = ap.parse_args()
    dict(plan=step_plan, draw=step_draw, need=step_need, unjudged=step_unjudged, score=step_score)[args.step](args)


if __name__ == "__main__":
    main()
