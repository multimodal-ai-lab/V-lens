"""Picturing a thought while copying a sentence (Sec. 4.4, the held-thoughts figure).

Setup of the intentional-control experiment of Lindsey (2026, arXiv:2601.01828): 'Write "{sentence}" exactly. Think
about {Thing} while you write. Don't write anything else.' The V-lens reads the tokens of the model's own copy of the
sentence. Items, conditions (think, mention, none) and the screen's layers are in configs/prompts/held_hunyuan.yaml.
HunyuanImage writes to $VLENS_OUT/held_hunyuan/, the other models to $VLENS_OUT/held_register_<model>/.

A trial is kept only if the copy is correct: case-insensitive Ratcliff-Obershelp similarity (difflib) >= 0.85 between
the reply and the sentence. A cell is a word or a low-content token (full stop, comma, determiner) of the copy and a
layer; in every trial it is read at the reply token covering that word's last letter (or at that token), so a cell means
the same position in the trials of both things.

reply (GPU): every item, thing and condition: the greedy reply and its similarity; at the end of the instruction and at
    every reply token, at every layer, the rank of each of the item's two names under the logit and the tuned lens
    (LENS_STORE/<model>_tuned_lens_n200, scripts/fit_tuned_lens.py); the states of those positions at every layer go to
    $VLENS_DATA/held_hunyuan/ (HunyuanImage) or $VLENS_DATA/held_register_<model>/.
select, register, dense (CPU, before any picture): screening cells (first and last word on a grid of layers plus the
    best tuned-lens cell; register tokens; every word and register token at seven layers).
cells (CPU, before any picture): a cell file for a confirmation run (--spec, --layers).
draw (GPU, sharded): V-lens pictures (coordinate-scaled copy into the insertion layer, "a photo of a thing") of the cells
    in --cells for the trials of --conditions, seeds --seeds; with --nouns also "a photo of a <thing>" for every thing.
judge (GPU): main subject among the item's two things, both, neither (three judges).
score (CPU): swap hits and, per token kind and layer, how often the picture shows the thought-of or the other thing.

The paper's two examples (confirmation cells, all kept trials of the item, seeds 0 to 2):

    # Show-o2: "the" in "After a pause, the progress continued." with a bicycle or a rose in mind, layers 22 to 26
    python scripts/held_thoughts.py reply --model show_o2
    python scripts/held_thoughts.py cells --model show_o2 --spec progress_short:the:1,progress_short:.:1 \\
        --layers 22,23,24,25,26 --cells cells_confirm.json
    python scripts/held_thoughts.py draw --model show_o2 --cells cells_confirm.json --conditions think,mention,none \\
        --seeds 0,1,2 --batch 8
    python scripts/held_thoughts.py judge --model show_o2 --judge qwen3vl   (and gemma3, llavaov2)
    python scripts/held_thoughts.py score --model show_o2

    # HunyuanImage: the final full stop of "The bus was crowded, but I found a seat near the back." with bread or an
    # elephant in mind, layers 23 to 27 (screen first: reply, register, draw --cells cells_register.json --seeds 0)
    python scripts/held_thoughts.py reply --model hunyuan
    python scripts/held_thoughts.py cells --model hunyuan --spec bus:.:1 --layers 23,24,25,26,27 \\
        --cells cells_confirm_bus.json
    python scripts/held_thoughts.py draw --model hunyuan --cells cells_confirm_bus.json --conditions think,mention,none \\
        --seeds 0,1,2
    python scripts/held_thoughts.py judge --model hunyuan --judge qwen3vl   (and gemma3, llavaov2)
    python scripts/held_thoughts.py score --model hunyuan
"""

import argparse
import difflib
import json
import math
import re
import time
import zlib
from collections import defaultdict

import torch
import yaml

from vlens import paths, prompts, results

OUT = results.RESULTS / "held_hunyuan"
STATES = paths.DATA / "held_hunyuan"
MODEL = "hunyuan"
#: dense cells' layers for the lighter models, each including the insertion layer
LAYERS = {"bagel": [4, 8, 12, 16, 20, 24, 27], "show_o2": [4, 8, 12, 16, 20, 24, 27],
          "sensenova": [6, 12, 18, 24, 30, 36, 41]}


def configure(model):
    global MODEL, OUT, STATES
    MODEL = model
    if model != "hunyuan":
        OUT = results.RESULTS / f"held_register_{model}"
        STATES = paths.DATA / f"held_register_{model}"
COPY_MIN = 0.85
PANEL = ("qwen3vl", "gemma3", "llavaov2")


def cfg():
    return yaml.safe_load((prompts.PROMPTS / "held_hunyuan.yaml").read_text())


def trials(c):
    """(trial id, item, condition, index of the thing in the prompt or None, prompt)."""
    out = []
    for it in c["items"]:
        for cond, template in c["templates"].items():
            for k in ([0, 1] if "{thing}" in template else [None]):
                text = template.format(sentence=it["sentence"], thing="" if k is None else it["things"][k])
                out.append((f"{it['id']}_{cond}" + ("" if k is None else f"_{k}"), it, cond, k, text))
    return out


def words(sentence):
    return re.findall(r"[A-Za-z']+", sentence)


TAG = re.compile(r"</?[a-z_]+>")   # HunyuanImage's control tags (</think>, <recaption>, ...), never part of the copy


def similarity(reply, sentence):
    a = TAG.sub("", reply).strip().strip('"“”').strip().lower()
    return difflib.SequenceMatcher(None, a, sentence.strip().lower()).ratio()


def word_tokens(strs, sentence):
    """For every word of the sentence, the index (into strs, the reply tokens) of the token covering its last letter in
    the reply, found in order and case-insensitively; None where the reply lacks the word."""
    starts, full = [], ""
    for s in strs:
        starts.append(len(full))
        full += s
    low, out, at = full.lower(), [], 0
    for w in words(sentence):
        hit = re.search(rf"(?<![a-z']){re.escape(w.lower())}(?![a-z'])", low[at:])
        if hit is None:
            out.append(None)
            continue
        end = at + hit.end() - 1
        out.append(max(i for i, s0 in enumerate(starts) if s0 <= end))
        at += hit.end()
    return out


REGISTER = (".", ",", "the", "a", "an")   # low-content tokens of the copy


def marks(r):
    """The copy's register tokens as {(token, occurrence): index into r["tokens"]}: full stops, commas and determiners,
    counted in order (a closing quote stuck to the full stop is ignored)."""
    out, seen = {}, defaultdict(int)
    for i, s in enumerate(r["tokens"][1:], start=1):
        t = s.strip().strip('"“”').strip().lower()
        if t in REGISTER:
            seen[t] += 1
            out[(t, seen[t])] = i
    return out


def position(r, cell):
    """The reply token a cell reads in trial r, or None."""
    if "word" in cell:
        return r["words"][cell["word"]]
    return marks(r).get((cell["mark"], cell["occ"]))


def cell_name(cell):
    return f"w{cell['word']:02d}" if "word" in cell else f"m{ {'.': 'stop', ',': 'comma'}.get(cell['mark'], cell['mark']) }{cell['occ']}"


def instruction_end(reading, first_reply):
    """The prompt's last text token: the full stop of "Don't write anything else." (or "else" itself)."""
    q = max(i for i in range(first_reply) if "else" in reading.token_strs[i])
    return q + 1 if q + 1 < first_reply and reading.token_strs[q + 1].strip() == "." else q


def step_reply(args):
    from vlens.models import load
    from vlens.paths import LENS_STORE
    from vlens.text_lens import rank
    c = cfg()
    m = load(MODEL)
    tuned = torch.load(LENS_STORE / f"{MODEL}_tuned_lens_n200" / "tuned.pt", map_location="cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    STATES.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    todo = [t for t in trials(c) if not args.items or t[1]["id"] in args.items.split(",")]
    for tid, it, cond, k, text in todo[args.shard::args.of]:
        if MODEL == "hunyuan":   # the closing tag is barred from the first position (an empty reply otherwise)
            reply, reading = m._follow(text, None, args.max_new_tokens, require_word=True)
        elif MODEL == "bagel":   # BAGEL's read_reply holds the reply only: the question's reading goes in front of it
            import dataclasses   # (read_reply itself is these two reads)
            reply = m.answer(text, max_new_tokens=args.max_new_tokens)
            first = m.read(text)
            rep = m.read(reply, previous=first.cache)
            reading = dataclasses.replace(
                rep, token_ids=first.token_ids + rep.token_ids, token_strs=first.token_strs + rep.token_strs,
                content=torch.cat([torch.zeros_like(first.content), rep.content]),
                hidden=[torch.cat([a, b]) for a, b in zip(first.hidden, rep.hidden)])
        else:
            reply = m.answer(text, max_new_tokens=args.max_new_tokens)
            reading = m.read_reply(text, reply)
        reply_pos = [q for q in range(len(reading.token_ids))
                     if reading.content[q] and not TAG.fullmatch(reading.token_strs[q].strip())]
        if not reply_pos:   # nothing written: kept as a failed copy
            rows.append(dict(trial=tid, item=it["id"], condition=cond, thing=k, prompt=text, reply=reply,
                             similarity=0.0, kept=False, tokens=[], words=[], ranks={}))
            print(f"[{time.time() - t0:5.0f}s] {tid}: empty reply {reply!r}", flush=True)
            continue
        positions = [instruction_end(reading, reply_pos[0])] + reply_pos
        strs = [reading.token_strs[q] for q in positions]
        ranks = defaultdict(list)
        for layer in range(m.n_layers):
            hs = torch.stack([reading.hidden[layer][q] for q in positions]).float().to(m.device)
            A, b = tuned["A"][layer].float().to(m.device), tuned["b"][layer].float().to(m.device)
            for lens, scores in (("logit", m.logit_lens(hs)), ("tuned", m.logit_lens(hs + hs @ A.T + b))):
                for n, name in enumerate(it["things"]):
                    ranks[f"{lens}_{n}"].append([rank(m, scores[i], name) for i in range(len(positions))])
        hidden = torch.stack([torch.stack([reading.hidden[l][q] for q in positions]) for l in range(m.n_layers)])
        torch.save(dict(positions=positions, hidden=hidden.half()), STATES / f"{tid}.pt")
        sim = similarity(reply, it["sentence"])
        wt = word_tokens(strs[1:], it["sentence"])
        rows.append(dict(trial=tid, item=it["id"], condition=cond, thing=k, prompt=text, reply=reply,
                         similarity=round(sim, 4), kept=sim >= COPY_MIN, tokens=strs,
                         words=[None if w is None else w + 1 for w in wt], ranks=ranks))
        print(f"[{time.time() - t0:5.0f}s] {tid}: {sim:.2f} {reply!r}", flush=True)
        tag = f"{args.shard}of{args.of}" + (f"_{args.tag}" if args.tag else "")
        (OUT / f"replies_{tag}.json").write_text(json.dumps(rows, indent=1))


def replies():
    rows = []
    for f in sorted(OUT.glob("replies_*.json")):
        rows += json.loads(f.read_text())
    return {r["trial"]: r for r in rows}


def step_select(args):
    """The first screen's cells, written before any picture. For every item whose two think trials are
    kept: (1) the grid, the sentence's first and last word at the layers grid_layers; (2) one extra cell, the word and
    layer (3 to 28, not in the grid) with the best symmetric tuned-lens score: for each thing k, log2 of its rank in the
    other think trial minus log2 of its rank in its own, the score being the smaller of the two (both swap directions
    must improve). Every cell is drawn in both think trials of its item."""
    c, rep = cfg(), replies()
    cells = []
    for it in c["items"]:
        t = [rep.get(f"{it['id']}_think_{k}") for k in (0, 1)]
        if not all(r is not None and r["kept"] for r in t):
            print(it["id"], "left out: a think copy is not kept")
            continue
        n = len(words(it["sentence"]))
        both = [w for w in range(n) if all(r["words"][w] is not None for r in t)]
        grid = [w for w in (0, n - 1) if w in both]
        cells += [dict(item=it["id"], word=w, layer=layer, kind="grid") for w in grid for layer in c["grid_layers"]]
        best = None
        for w in both:
            for layer in range(3, 29):
                if w in grid and layer in c["grid_layers"]:
                    continue
                d = [math.log2(t[1 - k]["ranks"][f"tuned_{k}"][layer][t[1 - k]["words"][w]])
                     - math.log2(t[k]["ranks"][f"tuned_{k}"][layer][t[k]["words"][w]]) for k in (0, 1)]
                if best is None or min(d) > best[0]:
                    best = (min(d), w, layer)
        if best is not None:
            cells.append(dict(item=it["id"], word=best[1], layer=best[2], kind="extra", score=round(best[0], 3)))
    (OUT / "cells_screen.json").write_text(json.dumps(cells, indent=1))
    print(len(cells), "cells ->", OUT / "cells_screen.json")


def step_register(args):
    """Register-token cells (punctuation and determiners), written before any of their pictures: the last full stop at layers 9, 16, 22, 25, 27, 29, and every other full stop, comma and
    determiner present in both think copies at layers 22 and 27. Drawn in both think trials."""
    c, rep = cfg(), replies()
    cells = []
    for it in c["items"]:
        if args.items and it["id"] not in args.items.split(","):
            continue
        t = [rep.get(f"{it['id']}_think_{k}") for k in (0, 1)]
        if not all(r is not None and r["kept"] for r in t):
            continue
        both = sorted(set(marks(t[0])) & set(marks(t[1])), key=lambda m: marks(t[0])[m])
        stops = [m for m in both if m[0] == "."]
        for m in both:
            layers = (9, 16, 22, 25, 27, 29) if stops and m == stops[-1] else (22, 27)
            cells += [dict(item=it["id"], mark=m[0], occ=m[1], layer=layer, kind="register") for layer in layers]
    name = "cells_register.json" if args.cells == "cells_screen.json" else args.cells   # --cells names a new file
    (OUT / name).write_text(json.dumps(cells, indent=1))
    print(len(cells), "cells ->", OUT / name)


def step_dense(args):
    """The lighter models' cells, written before any picture: in every item whose two think copies are kept, every word of
    the sentence and every register token present in both copies, at the seven layers of LAYERS."""
    c, rep = cfg(), replies()
    cells = []
    for it in c["items"]:
        t = [rep.get(f"{it['id']}_think_{k}") for k in (0, 1)]
        if not all(r is not None and r["kept"] for r in t):
            continue
        ws = [w for w in range(len(words(it["sentence"]))) if all(r["words"][w] is not None for r in t)]
        ms = sorted(set(marks(t[0])) & set(marks(t[1])), key=lambda m: marks(t[0])[m])
        for layer in LAYERS[MODEL]:
            cells += [dict(item=it["id"], word=w, layer=layer, kind="word") for w in ws]
            cells += [dict(item=it["id"], mark=m[0], occ=m[1], layer=layer, kind="register") for m in ms]
    (OUT / "cells_dense.json").write_text(json.dumps(cells, indent=1))
    print(len(cells), "cells ->", OUT / "cells_dense.json")


def step_cells(args):
    """A cell file for a confirmation run, written before its pictures: --spec is a comma-separated list of
    item:mark:occurrence (a register token of the copy, ".", ",", "the", "a" or "an", counted in order) or item:w<k> (the
    k-th word of the sentence, from 0), each at every layer of --layers."""
    items = {it["id"] for it in cfg()["items"]}
    layers = [int(x) for x in args.layers.split(",")]
    cells = []
    for spec in args.spec.split(","):
        parts = spec.split(":")
        if parts[0] not in items:
            raise ValueError(f"unknown item {parts[0]!r}")
        if len(parts) == 2 and parts[1].startswith("w"):
            cells += [dict(item=parts[0], word=int(parts[1][1:]), layer=l, kind="confirm") for l in layers]
        elif len(parts) == 3 and parts[1] in REGISTER:
            cells += [dict(item=parts[0], mark=parts[1], occ=int(parts[2]), layer=l, kind="confirm") for l in layers]
        else:
            raise ValueError(f"bad cell spec {spec!r}: item:mark:occurrence or item:w<k>")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.cells).write_text(json.dumps(cells, indent=1))
    print(len(cells), "cells ->", OUT / args.cells)


def step_draw(args):
    from vlens import readout
    from vlens.models import load
    c, rep = cfg(), replies()
    items = {it["id"]: it for it in c["items"]}
    cells = json.loads((OUT / args.cells).read_text())
    conditions = args.conditions.split(",")
    seeds = list(map(int, args.seeds.split(",")))
    units = []
    for cell in cells:
        for tid, r in sorted(rep.items()):
            if (r["item"] == cell["item"] and r["condition"] in conditions and r["kept"]
                    and (not args.trials or tid in args.trials.split(",")) and position(r, cell) is not None):
                units.append((tid, cell))
    m = load(MODEL)
    lens = readout.Lens(m, MODEL)
    out = OUT / MODEL
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.cells.replace('.json', '')}_{args.shard}of{args.of}"
    records, jobs = [], []
    if args.nouns and args.shard == 0:   # what each thing looks like when simply asked for
        for thing in sorted({t for it in c["items"] for t in it["things"]}):
            reading = m.read_prompt(f"a photo of a {thing.lower()}")
            for seed in seeds:
                f = f"noun_{thing.lower()}_s{seed}.png"
                jobs.append((f, reading, seed))
                others = [t for it in c["items"] for t in it["things"] if thing in it["things"] and t != thing]
                records.append(dict(file=f"{MODEL}/{f}", trial=None, item=None, condition="noun", thing=thing,
                                    word=None, layer=None, seed=seed, names=[thing, others[0]]))
    for tid, cell in units[args.shard::args.of]:
        r = rep[tid]
        st = torch.load(STATES / f"{tid}.pt")
        p = position(r, cell)
        reading = lens.reading(st["hidden"][cell["layer"]][p].float(), cell["layer"])
        for seed in seeds:
            f = f"{tid}_{cell_name(cell)}_L{cell['layer']:02d}_s{seed}.png"
            jobs.append((f, reading, seed))
            records.append(dict(file=f"{MODEL}/{f}", trial=tid, item=r["item"], condition=r["condition"],
                                thing=r["thing"], cell=cell_name(cell), word=cell.get("word"),
                                word_text=(words(items[r["item"]]["sentence"])[cell["word"]] if "word" in cell
                                           else cell["mark"]),
                                token=r["tokens"][p], layer=cell["layer"], kind=cell.get("kind"), seed=seed,
                                names=items[r["item"]]["things"]))
        if len(jobs) >= max(6, 4 * args.batch):
            readout.draw_all(m, jobs, out, args.batch)
            jobs.clear()
            (OUT / f"records_{tag}.json").write_text(json.dumps(records, indent=1))
    readout.draw_all(m, jobs, out, args.batch)
    (OUT / f"records_{tag}.json").write_text(json.dumps(records, indent=1))
    print(len(records), "pictures ->", out)


def records():
    rows = {}
    for f in sorted(OUT.glob("records_*.json")):
        for r in json.loads(f.read_text()):
            rows.setdefault(r["file"], r)   # a picture listed by two shards (a redraw) counts once
    return list(rows.values())


def step_judge(args):
    from PIL import Image
    from vlens.judges import Judge, subject_options, subject_question
    done = set()
    for f in OUT.glob(f"judge_{args.judge}_*.json"):
        done |= {d["file"] for d in json.loads(f.read_text())}
    todo = [r for r in records() if r["file"] not in done and (OUT / r["file"]).exists()
            and zlib.crc32(r["file"].encode()) % args.of == args.shard]
    print(f"{len(todo)} pictures to judge ({len(done)} before)", flush=True)
    j = Judge(args.judge)
    f = OUT / f"judge_{args.judge}_{args.shard}of{args.of}_{time.strftime('%H%M%S')}.json"
    ans = []
    for r in todo:
        try:   # a picture another job is still writing is left for the next judging round
            img = Image.open(OUT / r["file"])
            img.load()
        except OSError:
            print("not readable yet:", r["file"], flush=True)
            continue
        opts = subject_options(r["names"][0], r["names"][1], r["file"])
        ans.append(dict(file=r["file"], chosen=j.choose(img, opts, subject_question).chosen))
        f.write_text(json.dumps(ans, indent=1))
    print(len(ans), "->", f)


def step_score(args):
    """A picture shows a thing when at least 2 of 3 judges choose that thing alone. A cell (item, word, layer, seed) is a
    swap hit when the think trial of each thing shows that thing. Also listed: what the mention and none trials show at
    the same cell, and the plain drawings of the things."""
    votes = defaultdict(dict)
    for j in PANEL:
        for f in OUT.glob(f"judge_{j}_*.json"):
            for a in json.loads(f.read_text()):
                votes[a["file"]][j] = a["chosen"]
    shows = {}
    for r in records():
        v = list(votes.get(r["file"], {}).values())
        if len(v) == 3:
            top = max(set(v), key=v.count)
            shows[r["file"]] = top if v.count(top) >= 2 else None
    cell = defaultdict(dict)
    for r in records():
        if r["file"] in shows and r["condition"] != "noun":
            name = r.get("cell") or f"w{r['word']:02d}"
            cell[(r["item"], name, r["layer"], r["seed"])][f"{r['condition']}_{r['thing']}"] = shows[r["file"]]
    names = {it["id"]: it["things"] for it in cfg()["items"]}
    hits, n = [], 0
    for (item, w, layer, seed), got in sorted(cell.items()):
        if "think_0" in got and "think_1" in got:
            n += 1
            if got["think_0"] == names[item][0] and got["think_1"] == names[item][1]:
                hits.append(dict(item=item, cell=w, layer=layer, seed=seed, **got))
    nouns = {r["thing"]: shows.get(r["file"]) for r in records() if r["condition"] == "noun"}
    print("plain drawings shown as the thing:", sum(v == k for k, v in nouns.items()), "of", len(nouns),
          {k: v for k, v in nouns.items() if v != k})
    print(f"swap hits: {len(hits)} of {n} cells with both think pictures judged")
    for h in hits:
        print(" ", h)
    # Delimiters against words: in the think trials, how often the picture shows the thought-of
    # thing and how often the item's other thing, which the same sentence was copied with in the swapped trial
    table = defaultdict(lambda: [0, 0, 0])
    for r in records():
        if r["condition"] == "think" and r["file"] in shows:
            kind = "register" if r.get("cell", "w").startswith("m") else "word"
            for key in ((kind, r["layer"]), (kind, "all")):
                table[key][0] += 1
                table[key][1] += shows[r["file"]] == r["names"][r["thing"]]
                table[key][2] += shows[r["file"]] == r["names"][1 - r["thing"]]
    print("think pictures by token kind and layer: n, thought-of thing, other thing")
    for key in sorted(table, key=lambda k: (k[0], str(k[1]).zfill(3))):
        print(f"  {key[0]:8s} {str(key[1]):>3s}  {table[key][0]:5d}  {table[key][1]:4d}  {table[key][2]:4d}")
    (OUT / "score.json").write_text(json.dumps(dict(cells=n, hits=hits, nouns=nouns,
                                                    kinds={f"{k[0]}_{k[1]}": v for k, v in table.items()}), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["reply", "select", "register", "dense", "cells", "draw", "judge", "score"])
    ap.add_argument("--model", default="hunyuan", choices=["hunyuan", "show_o2", "sensenova", "bagel"])
    ap.add_argument("--batch", type=int, default=1, help="draw: pictures per batch (draw_many)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--items", default="", help="reply: only these item ids (comma separated)")
    ap.add_argument("--tag", default="", help="reply: suffix of the replies file (a second batch of items)")
    ap.add_argument("--cells", default="cells_screen.json", help="the cell file in the output folder")
    ap.add_argument("--spec", default="", help="cells: item:mark:occurrence or item:w<k>, comma separated")
    ap.add_argument("--layers", default="", help="cells: comma-separated layers")
    ap.add_argument("--conditions", default="think")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--trials", default="", help="draw: only these trial ids (comma separated), e.g. one thing's trial")
    ap.add_argument("--nouns", action="store_true")
    ap.add_argument("--judge", choices=PANEL)
    args = ap.parse_args()
    configure(args.model)
    dict(reply=step_reply, select=step_select, register=step_register, dense=step_dense, cells=step_cells, draw=step_draw,
         judge=step_judge, score=step_score)[args.step](args)


if __name__ == "__main__":
    main()
