"""Text lenses on the context-test items (Supp. D, the text-readout table): the ranks the text lenses give the two names. GPU.

At exactly the words and fixed extraction layers the context-test pictures were drawn from (configs/prompts/
followup_matched.yaml, followup_reference.yaml, followup_meanings.yaml; layers BAGEL 6, 8, 14; Show-o2 20, 24, 27;
SenseNova-U1 20, 24, 32; HunyuanImage-3.0 9, 12, 15), up to three text readouts give a ranked list of words: the logit
lens (final norm and text head on the state as it is), the tuned lens (scripts/fit_tuned_lens.py) and the text J-lens
(`<model>_text_jlens_n200`, scripts/fit_text_jlens.py), each when fitted. HunyuanImage-3.0 has no J-lens (its fit would
need one backward pass per hidden dimension through 80B parameters). For every text: the rank of its own thing and of
the pair's other thing (the best first token of either name's capitalization and spacing variants), and whether the two
names start with different tokens (only such texts are scored by scripts/score_text_lenses.py).
Writes $VLENS_OUT/followup_cells/text_lenses_<model>.json.

    python scripts/text_lens_ranks.py --model bagel --set followup     # also show_o2, sensenova, hunyuan
"""

import argparse
import json

import torch
import yaml

from vlens import cells, prompts, results
from vlens.models import load
from vlens.paths import LENS_STORE
from vlens.text_lens import TextLens, rank, word_ids

BAND = {"bagel": [6, 8, 14], "show_o2": [20, 24, 27], "sensenova": [20, 24, 32], "hunyuan": [9, 12, 15]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="bagel")
    ap.add_argument("--set", default="followup", choices=["followup"],
                    help="the context-test items at their fixed extraction layers")
    args = ap.parse_args()
    m = load(args.model)
    # no text J-lens for HunyuanImage-3.0: logit and tuned lens only
    lens = TextLens(f"{args.model}_text_jlens_n200", device=m.device) if (LENS_STORE / f"{args.model}_text_jlens_n200").exists() else None
    tf = LENS_STORE / f"{args.model}_tuned_lens_n200" / "tuned.pt"
    tuned = torch.load(tf, map_location=m.device) if tf.exists() else None
    grid, folder = BAND, "followup_cells"
    jobs = [("followup_matched", "reads", None), ("followup_reference", "reads", None),
            ("followup_meanings", "double_meaning", None)]
    meta = {n: yaml.safe_load((prompts.PROMPTS / f"{n}.yaml").read_text())["reads"] for n, k, _ in jobs if k == "reads"}
    rows = []
    with torch.no_grad():
        for items, kind, only in jobs:
            for rec in cells.texts(items, kind, only):
                if items in meta and meta[items][rec["item"]]["role"] not in ("text1", "text2"):
                    continue
                src = m.read(rec["text"])
                for pos_name, word, occurrence, truth in rec["positions"]:
                    if items in meta:
                        other = meta[items][rec["item"]]["alt"]
                    else:
                        it = yaml.safe_load((prompts.PROMPTS / f"{items}.yaml").read_text())["double_meaning"][rec["item"]]
                        names = [x["name"] if isinstance(x, dict) else x for x in (it["meaning_a"], it["meaning_b"])]
                        other = next(n for n in names if n != truth)
                    distinct = not (set(word_ids(m.tokenizer, truth)) & set(word_ids(m.tokenizer, other)))
                    pos = cells.word_tokens(src, word, occurrence)[-1]
                    for layer in grid[args.model]:
                        h = src.hidden[layer][pos][None]
                        reads = {"logit": m.logit_lens(h)[0]}
                        if lens is not None:
                            reads["jlens"] = m.logit_lens(lens.transport(h, layer).to(h.dtype))[0]
                        if tuned is not None:
                            hf = h.float()
                            t = hf + hf @ tuned["A"][layer].float().to(hf.device).T + tuned["b"][layer].float().to(hf.device)
                            reads["tuned"] = m.logit_lens(t.to(h.dtype))[0]
                        r = dict(items=items, item=rec["item"], order=rec.get("order"), position=pos_name, word=word,
                                 truth=truth, other=other, distinct=distinct, layer=layer)
                        for k, s in reads.items():
                            r[f"{k}_rank"] = rank(m, s, truth)
                            r[f"{k}_rank_other"] = rank(m, s, other)
                        rows.append(r)
                print(f"{items} {rec['item']} done", flush=True)
    out = results.folder(folder) / f"text_lenses_{args.model}.json"
    out.write_text(json.dumps(rows, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
