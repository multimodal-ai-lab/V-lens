"""The described object across tokens and layers in the two autoregressive models, Liquid and Janus-Pro (the Liquid
panel of Fig. 4, Sec. 4.3).

The same items, positions, layers, seeds, judges and rules as depth_fixed_positions.py and depth_through_answer.py, in
one script:
- answers (GPU, one job per model): the 16 two-hop questions of configs/prompts/bridges.yaml (as written), each asked
  with " Answer with one word." in the model's chat format, greedy reply (the adapter's own answer(), with the reply's
  token ids kept). A question counts when the reply is right and does not name the object before the answer. For every
  counted question the positions to read: the description's last word and every later token through the answer's first
  token (question words, question mark, instruction, chat-format tokens, the token before the answer, the answer
  token), all from one reading of prompt and reply (teacher forcing). Named references: the 15 objects in
  "Yesterday I saw a {object} on my way home."
- draw (GPU, sharded): the V-lens (vlens.readout.Lens: the coordinate scaling of Eq. 1 into "a photo of a thing" at the
  insertion layer, Liquid 6, Janus-Pro 8) of each position at every second layer and the last, drawing seeds 3, 4, 5.
- judged by the three judges with the main-subject question (run_judge.py --set cells:depth_autoregressive --subject).
- score (CPU): per model and layer, the shares of counted questions at the description word and at the best later
  token, and of the 15 named references, as Fig. 4 plots them; and which kinds of positions carry the later hits.

Pictures and records in $VLENS_OUT/depth_autoregressive/<model>/.

    python scripts/depth_autoregressive.py answers --model liquid
    python scripts/depth_autoregressive.py draw --model liquid --shard 0 --of 100
    python scripts/run_judge.py --judge qwen3vl --set cells:depth_autoregressive --subject --model liquid   (and gemma3, llavaov2)
    python scripts/depth_autoregressive.py score
"""

import argparse
import json
import time
from collections import defaultdict

import torch
import yaml

from vlens import prompts, readout, results
from vlens.judges import answer_files

OUT = results.RESULTS / "depth_autoregressive"
JUDGES = results.RESULTS / "judges"
FOLDER = "depth_autoregressive"
SEEDS = (3, 4, 5)
MODELS = ["liquid", "janus_pro"]
PANEL = ["qwen3vl", "gemma3", "llavaov2"]



def items():
    reads = yaml.safe_load((prompts.PROMPTS / "bridges.yaml").read_text())
    qs = [(i, it) for i, it in enumerate(reads["reads"]) if it["family"] in ("bridge", "same_answer")]
    truths = {it["truth"] for _, it in qs}
    ctl = [(j, c) for j, c in enumerate(reads["controls"]) if c["object"] in truths]
    return qs, ctl


# --- model access ----------------------------------------------------------------------------------------------------
def prompt_ids(m, model, asked):
    if model == "liquid":
        return m._ids(m.chat(asked))[0].tolist()
    return m.tokenizer.encode(m._format(asked, False))


@torch.no_grad()
def generate(m, model, asked):
    """The greedy one-word reply exactly as the adapter's answer() makes it, with its token ids."""
    ids = prompt_ids(m, model, asked)
    x = torch.tensor([ids], device=m.device)
    if model == "liquid":
        out = m.net.generate(input_ids=x, max_new_tokens=readout.MAX_REPLY, do_sample=False,
                             eos_token_id=[m.tokenizer.eos_token_id, m.tokenizer.convert_tokens_to_ids("<end_of_turn>")])
        new = out[0, x.shape[1]:].tolist()
    else:
        out = m.lm.generate(inputs_embeds=m.lm.get_input_embeddings()(x), attention_mask=torch.ones_like(x),
                            pad_token_id=m.tokenizer.eos_token_id, bos_token_id=m.tokenizer.bos_token_id,
                            eos_token_id=m.tokenizer.eos_token_id, max_new_tokens=readout.MAX_REPLY, do_sample=False)
        new = out[0].tolist()
    text = m.tokenizer.decode(new, skip_special_tokens=True).strip()
    assert text == m.answer(asked), (text, m.answer(asked))
    return ids, new, text


@torch.no_grad()
def capture(m, model, ids):
    """Hidden states entering every layer for one token sequence (the adapters' capture hooks)."""
    m._mode, m._capture = "capture", [None] * m.n_layers
    try:
        x = torch.tensor([ids], device=m.device)
        if model == "liquid":
            m.net(input_ids=x)
        else:
            m.lm.model(inputs_embeds=m.lm.get_input_embeddings()(x))
    finally:
        m._mode = None
    return list(m._capture)


def spans(strs):
    starts, full = [], ""
    for s in strs:
        starts.append(len(full))
        full += s
    return starts, full


# --- answers and positions -------------------------------------------------------------------------------------------
def step_answers(args):
    from vlens.models import load
    m = load(args.model)
    qs, ctl = items()
    rows = []
    for i, it in enumerate_qs(qs):
        asked = it["text"] + readout.ONE_WORD
        p_ids, r_ids, reply = generate(m, args.model, asked)
        hit = readout.answer_hit(reply, it["answers"])
        before = hit is not None and readout.answer_hit(reply[: hit[0]], readout.names(it)) is not None
        row = dict(item=i, id=it["id"], family=it["family"], truth=it["truth"], alt=it["alt"], reply=reply,
                   right=hit is not None, bridge_before_answer=before, counted=hit is not None and not before,
                   prompt_ids=p_ids, reply_ids=r_ids)
        if row["counted"]:
            row["positions"] = positions(m, args.model, it, p_ids, r_ids)
        rows.append(row)
        print(f"{it['id']}: {reply!r} right={row['right']} counted={row['counted']}"
              + (f" positions={len(row['positions'])}" if row["counted"] else ""), flush=True)
    refs = []
    for j, c in ctl:
        src = m.read(c["text"])
        refs.append(dict(item=j, object=c["object"], alt=c["literals"][0], token_ids=list(src.token_ids),
                         position=readout.last_word(src, c["object"]), token=src.token_strs[readout.last_word(src, c["object"])]))
    # timing of one drawing, for the shard plan
    lens = readout.Lens(m, args.model)
    t0 = time.time()
    m.draw(lens.reading(capture(m, args.model, refs[0]["token_ids"])[m.n_layers // 2][refs[0]["position"]],
                        m.n_layers // 2), seed=3)
    sec = time.time() - t0
    out = OUT / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.json").write_text(json.dumps(dict(model=args.model, n_layers=m.n_layers, R=lens.R,
                                                      layers=readout.layers(m.n_layers), seconds_per_picture=round(sec, 1),
                                                      questions=rows, references=refs), indent=1))
    n_pos = sum(len(r.get("positions", [])) for r in rows)
    print(f"{sum(r['counted'] for r in rows)} of {len(rows)} questions counted, {n_pos} positions, "
          f"{len(refs)} references, {sec:.1f} s per picture -> {out / 'answers.json'}")


def enumerate_qs(qs):
    return [(i, it) for i, it in qs]


def positions(m, model, it, p_ids, r_ids):
    """Every token from the description's last word through the answer's first token, with its kind."""
    tok = m.tokenizer
    p_strs = [tok.decode([t]) for t in p_ids]
    r_strs = [tok.decode([t]) for t in r_ids]
    p_starts, p_full = spans(p_strs)
    r_starts, r_full = spans(r_strs)
    # the answer's first token in the reply (its text read as the tokens decode it)
    hit = readout.answer_hit(r_full, it["answers"])
    assert hit is not None, (it["id"], r_full)
    k = [n for n, s0 in enumerate(r_starts) if s0 + len(r_strs[n]) > hit[0]][0]
    answer_at = len(p_ids) + k
    # the description word: its last token, read in the prompt only (the reply may restate words)
    from vlens.models.base import TextReading
    desc = readout.last_word(TextReading(text="", token_ids=p_ids, token_strs=p_strs, hidden=[], content=None, cache={}),
                          it["positions"]["desc"])
    q0 = p_full.find(it["text"])
    assert q0 >= 0, (it["id"], p_full)
    q1 = q0 + len(it["text"])
    i0 = p_full.find(readout.ONE_WORD.strip(), q1)
    assert i0 >= 0
    i1 = i0 + len(readout.ONE_WORD.strip())
    out = []
    strs = p_strs + r_strs[: k + 1]
    starts = p_starts + [len(p_full) + s for s in r_starts[: k + 1]]
    for p in range(desc, answer_at + 1):
        a, b = starts[p], starts[p] + len(strs[p])
        if p == desc:
            kind = "desc"
        elif p == answer_at:
            kind = "answer"
        elif p == answer_at - 1:
            kind = "preanswer"
        elif p < len(p_ids) and a < q1 and b > q0:
            kind = "qmark" if strs[p].strip() == "?" else "question"
        elif p < len(p_ids) and a < i1 and b > i0:
            kind = "instruction"
        else:
            kind = "chat"
        out.append(dict(index=p, token=strs[p], kind=kind))
    return out


# --- drawing ---------------------------------------------------------------------------------------------------------
def units(plan):
    """One unit = one position of one counted question, or one named reference; the order fixes the shards."""
    u = [("ref", r["item"], None) for r in plan["references"]]
    for q in plan["questions"]:
        for pos in q.get("positions", []):
            u.append(("q", q["item"], pos["index"]))
    return u


def step_draw(args):
    from vlens.models import load
    out = OUT / args.model
    plan = json.loads((out / "answers.json").read_text())
    todo = units(plan)[args.shard::args.of]
    L = plan["layers"]
    m = load(args.model)
    lens = readout.Lens(m, args.model)
    assert lens.R == plan["R"]
    qs = {q["item"]: q for q in plan["questions"]}
    refs = {r["item"]: r for r in plan["references"]}
    records, t0, tag = [], time.time(), f"{args.shard}of{args.of}"
    rec_file = out / f"records_{tag}.json"
    states = {}
    for kind, item, idx in todo:
        if kind == "ref":
            r = refs[item]
            key = ("ref", item)
            if key not in states:
                states = {key: capture(m, args.model, r["token_ids"])}
            hidden, p = states[key], r["position"]
            base = dict(items="bridges", kind="controls", item=item, position="word", token=r["token"],
                        truth=r["object"], alt=r["alt"])
            stem = f"ref_{r['object'].replace(' ', '-')}"
        else:
            q = qs[item]
            key = ("q", item)
            if key not in states:
                states = {key: capture(m, args.model, q["prompt_ids"] + q["reply_ids"])}
            hidden, p = states[key], idx
            pos = next(x for x in q["positions"] if x["index"] == idx)
            base = dict(items="bridges", kind="reads", item=item, family=q["family"], condition="inferred",
                        position=f"p{idx:03d}", pos_kind=pos["kind"], token=pos["token"], truth=q["truth"], alt=q["alt"])
            stem = f"{q['id']}_p{idx:03d}"
        for l in L:
            reading = lens.reading(hidden[l][p], l)
            for seed in SEEDS:
                f = f"{stem}_L{l:02d}_s{seed}.png"
                if not (out / f).exists():
                    m.to_picture(m.draw(reading, seed=seed)[0]).save(out / f)
                records.append(dict(base, file=f"{args.model}/{f}", model=args.model, layer=l, way="lens", seed=seed))
        rec_file.write_text(json.dumps(records, indent=1))
        print(f"[{time.time() - t0:6.0f}s] {stem}: {len(records)} pictures", flush=True)
    rec_file.write_text(json.dumps(records, indent=1))


def step_missing(args):
    """Pictures still missing (file or record), per shard of a plan: "<shard> <missing>" lines."""
    for model in MODELS if args.model is None else [args.model]:
        out = OUT / model
        if not (out / "answers.json").exists():
            print(f"{model} no-plan")
            continue
        plan = json.loads((out / "answers.json").read_text())
        have = set()
        for f in out.glob("records_*.json"):
            try:
                have |= {r["file"] for r in json.loads(f.read_text())}
            except json.JSONDecodeError:
                pass
        files = {f.name for f in out.glob("*.png")}
        qs = {q["item"]: q for q in plan["questions"]}
        refs = {r["item"]: r for r in plan["references"]}
        u = units(plan)
        for s in range(args.of):
            n = 0
            for kind, item, idx in u[s::args.of]:
                stem = f"ref_{refs[item]['object'].replace(' ', '-')}" if kind == "ref" else f"{qs[item]['id']}_p{idx:03d}"
                for l in plan["layers"]:
                    for seed in SEEDS:
                        f = f"{stem}_L{l:02d}_s{seed}.png"
                        n += not (f in files and f"{model}/{f}" in have)
            print(f"{model} {s} {n}")


def step_unjudged(args):
    files = set()
    for model in MODELS:
        for f in (OUT / model).glob("records_*.json"):
            files |= {r["file"] for r in json.loads(f.read_text())}
    done = set()
    for f in answer_files(JUDGES, args.judge, f"cells:{FOLDER}_subject"):
        done |= {a["file"] for a in json.loads(f.read_text())}
    print(len(files - done))


# --- scoring ---------------------------------------------------------------------------------------------------------
def step_score(args):
    votes = defaultdict(dict)
    for j in PANEL:
        for f in answer_files(JUDGES, j, f"cells:{FOLDER}_subject"):
            for a in json.loads(f.read_text()):
                votes[a["file"]][j] = a["chosen"]
    out_all = {}
    for model in MODELS:
        out = OUT / model
        if not (out / "answers.json").exists():
            continue
        plan = json.loads((out / "answers.json").read_text())
        recs = {}
        for f in out.glob("records_*.json"):
            for r in json.loads(f.read_text()):
                recs[r["file"]] = r
        cells = defaultdict(list)
        n_missing = 0
        for r in recs.values():
            v = votes.get(r["file"], {})
            if len(v) < len(PANEL):
                n_missing += 1
                continue
            shows = sum(c == r["truth"] for c in v.values()) >= 2
            cells[(r["kind"], r["item"], r["position"], r["layer"])].append(shows)
        cell = {k: sum(v) >= 2 for k, v in cells.items() if len(v) == 3}
        counted = [q for q in plan["questions"] if q["counted"]]
        rows = {}
        for l in plan["layers"]:
            desc = later = 0
            kinds = defaultdict(int)
            complete = True
            for q in counted:
                ps = q["positions"]
                got = {p["index"]: cell.get(("reads", q["item"], f"p{p['index']:03d}", l)) for p in ps}
                if any(v is None for v in got.values()):
                    complete = False
                d = ps[0]
                desc += bool(got[d["index"]])
                lat = [p for p in ps[1:] if got[p["index"]]]
                later += bool(lat)
                for k in {p["kind"] for p in lat}:
                    kinds[k] += 1
            named = sum(bool(cell.get(("controls", r["item"], "word", l))) for r in plan["references"])
            rows[l] = dict(desc=desc, best_later=later, named=named, of=len(counted), of_named=len(plan["references"]),
                           later_by_kind=dict(kinds), complete=complete)
        out_all[model] = dict(R=plan["R"], counted=len(counted), questions=len(plan["questions"]),
                              unjudged_pictures=n_missing, layers=rows,
                              replies={q["id"]: q["reply"] for q in plan["questions"]})
        print(f"\n{model} (R = {plan['R']}): {len(counted)} of {len(plan['questions'])} questions counted; "
              f"{n_missing} pictures not judged by all three")
        print("layer  description  best later  named   later kinds")
        for l, r in rows.items():
            print(f"{l:5d}  {r['desc']:>5}/{r['of']:<5} {r['best_later']:>5}/{r['of']:<5} {r['named']:>3}/{r['of_named']:<3}"
                  f"  {r['later_by_kind']}{'' if r['complete'] else '  (incomplete)'}")
    results.save_numbers(OUT / "scores.json", out_all)
    print("->", OUT / "scores.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["answers", "draw", "missing", "unjudged", "score"])
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--judge")
    args = ap.parse_args()
    {"answers": step_answers, "draw": step_draw, "missing": step_missing, "unjudged": step_unjudged,
     "score": step_score}[args.step](args)
