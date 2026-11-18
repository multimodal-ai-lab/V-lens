"""Text readouts of the depth study's states (Supp. D, the text-readout table; Sec. 4.3).

At exactly the states the depth figure draws (Fig. 4): the description's last word, and every later token through the
first answer token (the question's tokens after the describing word from the plain reading of
scripts/depth_fixed_positions.py and scripts/depth_all_tokens.py; the instruction, chat-format, pre-answer and first
answer tokens from the teacher-forced answer reading of scripts/depth_through_answer.py, its plan.json), at the same
layers, three text readouts rank the vocabulary: the logit lens, the tuned lens (scripts/fit_tuned_lens.py) and the text
J-lens (scripts/fit_text_jlens.py), each when fitted (HunyuanImage-3.0 has no J-lens). Saved per (question, position,
layer, readout): the rank of the object's name and of the alternative's name (the best first token of their spellings,
vlens.text_lens.rank).

score counts, per model, readout and layer, the questions whose object is ranked first (top1), among the ten most likely
tokens (top10; the table's criterion) or above the alternative's name (above_alt), at the description's last word and at
the best later token, next to the pictures' counts of Fig. 4 (scripts/depth_fixed_positions.py score and
scripts/depth_through_answer.py score must have run). Writes $VLENS_OUT/text_readout_depth/<model>/ranks.json and
$VLENS_OUT/text_readout_depth/scores.json.

    python scripts/text_readout_depth.py read --model bagel     # GPU; also show_o2, sensenova, hunyuan (four GPUs)
    python scripts/text_readout_depth.py score                  # CPU
"""

import argparse
import json
from collections import defaultdict

from vlens import results

OUT = results.RESULTS / "text_readout_depth"
ANS = results.RESULTS / "alltokens_answer"
FIG = {"confirm": results.RESULTS / "bridge_confirm" / "scores.json", "later": ANS / "scores.json"}
MODELS = ("bagel", "show_o2", "sensenova", "hunyuan")
R = {"bagel": 8, "show_o2": 27, "sensenova": 24, "hunyuan": 9}
CRITERIA = {"top1": lambda o, a: o == 1, "top10": lambda o, a: o <= 10, "above_alt": lambda o, a: o < a}


def step_read(args):
    import torch
    from vlens import readout
    from vlens.hidden import find_word
    from vlens.models import load
    from vlens.paths import LENS_STORE
    from vlens.text_lens import TextLens, rank
    through = results.load_script("depth_through_answer")
    fixed = through.fixed
    model = args.model
    qs = dict(fixed.items()[0])
    plan = json.loads((ANS / model / "plan.json").read_text())
    use, ans, pre_fixed = through.recorded(model)
    assert [q["item"] for q in plan["questions"]] == list(use), "plan.json and the counted questions differ"
    m = load(model)
    jl = TextLens(f"{model}_text_jlens_n200", device=m.device) if (LENS_STORE / f"{model}_text_jlens_n200").exists() else None
    tf = LENS_STORE / f"{model}_tuned_lens_n200" / "tuned.pt"
    tuned = torch.load(tf, map_location=m.device) if tf.exists() else None
    tcache = {}

    def tuned_at(layer, device):   # the tuned lens's affine map of one layer, in float32 on the state's device
        if layer not in tcache:
            tcache[layer] = (tuned["A"][layer].float().to(device).T, tuned["b"][layer].float().to(device))
        return tcache[layer]
    L = plan["layers"]
    print(f"{model}: {len(use)} questions, layers {L}, J-lens {jl is not None}, tuned lens {tuned is not None}", flush=True)

    def ranks(h, layer, names, alt):
        """{readout: (rank of the best of names, rank of alt)} for one state h [1, d]."""
        reads = {"logit": m.logit_lens(h)[0]}
        if jl is not None:
            reads["jlens"] = m.logit_lens(jl.transport(h, layer).to(h.dtype))[0]
        if tuned is not None:
            hf = h.float()
            At, b = tuned_at(layer, hf.device)
            reads["tuned"] = m.logit_lens((hf + hf @ At + b).to(h.dtype))[0]
        return {k: (min(rank(m, s, n) for n in names), rank(m, s, alt)) for k, s in reads.items()}

    rows = []
    with torch.no_grad():
        for q in plan["questions"]:
            i, it = q["item"], qs[q["item"]]
            names, alt = readout.names(it), it["alt"]
            # the plain reading the fixed-position and all-tokens pictures were drawn from
            src = through.plain_reading(m, model, it["text"])
            content = [p for p in range(len(src.token_ids)) if src.content[p]]
            last = readout.last_word(src, it["positions"]["desc"])
            assert last in set(find_word(src, it["positions"]["desc"]))
            spots = [("desc", "desc", src, last)]
            spots += [(f"tok{p:02d}", "question mark" if p == content[-1] else "question word", src, p)
                      for p in content if p > last]
            assert src.token_strs[content[-1]].strip().endswith("?"), src.token_strs[content[-1]]
            # the teacher-forced answer reading (the same ids the through-answer pictures were drawn from)
            seq, _ = through.answer_reading(m, model, q["asked"], q["reply"], q.get("gen_ids"))
            assert seq.ids[:q["ans"] + 1] == q["ids"], f"{q['id']}: answer reading differs from plan.json"
            spots += [(f"post{x['p']:03d}", x["group"], seq, x["p"]) for x in q["positions"]]
            spots += [("preanswer", "pre-answer", seq, q["pre"]), ("answer1", "answer token", seq, q["ans"])]
            assert seq.strs[q["pre"]] == pre_fixed[i]
            for pos, kind, rd, p in spots:
                plain = hasattr(rd, "token_strs")
                tok = rd.token_strs[p] if plain else rd.strs[p]
                for layer in L:
                    h = (rd.hidden[layer][p] if plain else rd.hidden(layer, p))[None]
                    for lens, (own, other) in ranks(h, layer, names, alt).items():
                        rows.append(dict(item=i, id=it["id"], family=it["family"], truth=it["truth"], alt=alt,
                                         position=pos, kind=kind, token=tok, layer=layer, lens=lens, rank=own,
                                         rank_alt=other))
            best = min(r["rank"] for r in rows if r["item"] == i and r["position"] == "desc")
            print(f"{q['id']:16s} {len(spots)} positions; best rank at the describing word {best}", flush=True)
            torch.cuda.empty_cache()
    out = results.folder("text_readout_depth") / model
    out.mkdir(parents=True, exist_ok=True)
    (out / "ranks.json").write_text(json.dumps(rows))
    print(f"-> {out / 'ranks.json'} ({len(rows)} rows)", flush=True)


def frac(s):
    a, b = s.split("/")
    return int(a), int(b)


def step_score(args):
    confirm, later = json.loads(FIG["confirm"].read_text()), json.loads(FIG["later"].read_text())
    out, lines = {}, []
    for model in MODELS:
        f = OUT / model / "ranks.json"
        if not f.exists():
            lines.append(f"{model}: no ranks yet")
            continue
        rows = json.loads(f.read_text())
        items = sorted({r["item"] for r in rows})
        layers = sorted({r["layer"] for r in rows})
        pic_desc = {int(l): frac(v)[0] for l, v in confirm[model]["curves"]["inferred|desc"].items()}
        pic_later = {int(l): int(v) for l, v in later[model]["best_later"].items()}
        res = dict(counted=len(items), layers=layers, pictures=dict(desc=pic_desc, best_later=pic_later), text={})
        cell = defaultdict(lambda: defaultdict(bool))   # (readout, criterion, where, layer) -> item -> hit
        for r in rows:
            where = "desc" if r["position"] == "desc" else "best_later"
            for c, fn in CRITERIA.items():
                cell[(r["lens"], c, where, r["layer"])][r["item"]] |= fn(r["rank"], r["rank_alt"])
        for (lens, c, where, layer), d in cell.items():
            res["text"].setdefault(lens, {}).setdefault(c, {}).setdefault(where, {})[layer] = sum(d.values())
        out[model] = res
        n, r_ = len(items), R[model]
        lines.append(f"\n{model} (n = {n}, R = {r_}); pictures at R: description {pic_desc.get(r_)}/{n}, best later "
                     f"{pic_later.get(r_)}/{n}; best layer: description {max(pic_desc.values())}/{n}, best later "
                     f"{max(pic_later.values())}/{n}")
        for lens in res["text"]:
            for c in CRITERIA:
                for where in ("desc", "best_later"):
                    v = res["text"][lens][c][where]
                    bl = max(v, key=lambda l: (v[l], l))
                    lines.append(f"  {lens:6s} {c:9s} {where:10s} at R {v[r_]:2d}/{n}   best {v[bl]:2d}/{n} (layer {bl})")
    results.folder("text_readout_depth")
    (OUT / "scores.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["read", "score"])
    ap.add_argument("--model", default="bagel", choices=MODELS)
    args = ap.parse_args()
    {"read": step_read, "score": step_score}[args.step](args)


if __name__ == "__main__":
    main()
