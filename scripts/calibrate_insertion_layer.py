"""Choose a model's insertion layer R (Sec. 3, Supp. A.5, Supp. Table S4; Supp. I for Liquid and Janus-Pro), with a few
checks of the model adapter.

Part A (shard 0 only), adapter checks:
  1. replacing the slot "thing" of "a photo of a thing" by its own hidden state leaves the picture unchanged;
  2. text answers to the 16 two-hop questions of configs/prompts/vlens_items.yaml;
  3. normal drawings of a few prompts, with the time per picture.

Part B, the insertion layer: copy the hidden state entering layer l at one token of a source text into the slot "thing"
of "a photo of a thing" at the same layer, draw (the model's own settings, seed 0), and let CLIP ViT-L/14 rank the
concept among the 100 concept names (configs/prompts/concepts.yaml). Sources: the concept word in "Yesterday I saw a
{word} on my way home." (the paper's calibration), the last word of a one-hop description (configs/prompts/ladder.yaml),
and the control word "the". 12 words: spider, apple, banana, penguin, pyramid, piano, elephant, rocket, kangaroo,
octopus, bee, camel.

Rule: R is the latest tested layer at which the plain-sentence copy is CLIP's first choice for the most words
(scripts/lens_stats_own_tokens.py reads it back for HunyuanImage, Liquid and Janus-Pro). Layers: about a dozen from 0
to the last (default); Table S4 used --layers 0,2,4,8,12,16,20,24,27 for BAGEL and
--layers 0,2,4,6,8,10,12,14,16,20,24,27 for Show-o2, and the default for SenseNova-U1, HunyuanImage, Liquid and
Janus-Pro. Writes $VLENS_OUT/check_<model name>/numbers_shard<i>of<n>.json and picture grids.

    python scripts/calibrate_insertion_layer.py --model sensenova --shard 0 --of 6     # shards 0..5
    python scripts/calibrate_insertion_layer.py --model bagel --layers 0,2,4,8,12,16,20,24,27 --shard 0 --of 4
"""

import argparse
import time

import torch
import yaml

from vlens import gallery, prompts, results, score
from vlens.hidden import find_word, replace_at
from vlens.models import load, no_system_message

CONCEPTS = ["spider", "apple", "banana", "penguin", "pyramid", "piano", "elephant", "rocket", "kangaroo", "octopus",
            "bee", "camel"]
TARGET = "a photo of a thing"
SIZES = {"sensenova": (512, 1024)}


def scan_layers(n_layers: int) -> list[int]:
    """About a dozen layers from 0 to the last, evenly spread."""
    step = max(1, round(n_layers / 12))
    return sorted(set(list(range(0, n_layers, step)) + [n_layers - 1]))


def sources_for(concept: str) -> dict:
    one = next(i for i in prompts.ladder("level3_one_hop") if i["concept"] == concept)["prompt"]
    return {"plain sentence": (f"Yesterday I saw a {concept} on my way home.", concept),
            "one-hop description": (one, None),
            "control: 'the'": ("Yesterday I saw the sky on my way home.", "the")}


def part_a(m, model, out, numbers, size):
    target = m.read_prompt(TARGET)
    slot = find_word(target, "thing")[-1]
    plain = m.draw(target, size=size, seed=0)[0]
    same = m.draw(m.read_prompt(TARGET, edit=replace_at(m.n_layers // 4, [slot], target.hidden[m.n_layers // 4][slot][None])),
                  size=size, seed=0)[0]
    numbers["identity_edit_max_abs"] = float((plain.float() - same.float()).abs().max())

    data = yaml.safe_load((prompts.PROMPTS / "vlens_items.yaml").read_text())
    try:
        numbers["two_hop_answers"] = [dict(prompt=it["prompt"], expected=it["answer"], reply=m.answer(it["prompt"]))
                                      for it in data["two_hop"]]
    except Exception as e:  # a failing text reply must not cost the drawings and the scan
        numbers["two_hop_answers"] = [dict(error=repr(e))]
    for a in numbers["two_hop_answers"]:
        print(a, flush=True)

    texts = ["a photo of a spider", TARGET, "", "How many legs does the animal that spins webs have?",
             "a photo of the animal that spins webs", "a photo of an elephant in the snow"]
    rows = []
    for s in SIZES.get(model, (size,)):
        t0 = time.time()
        rows.append([m.to_picture(m.draw(m.read_prompt(t), size=s, seed=0)[0]) for t in texts])
        numbers[f"seconds_per_picture_{s}px"] = (time.time() - t0) / len(texts)
    gallery.grid(rows, row_labels=[f"{s} px" for s in SIZES.get(model, (size,))],
                 col_labels=[t[:24] or "(empty)" for t in texts], cell=200,
                 title=f"{m.name}, normal drawing (seed 0)").save(out / "drawings.png")


def main():
    ap = argparse.ArgumentParser(description="Choose the insertion layer by copying word states into the drawing slot.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--size", type=int, default=512)
    # for a slow model (HunyuanImage, minutes per picture): the words to scan, sharded with --shard/--of, which sources,
    # and whether to skip part A on shard 0
    ap.add_argument("--layers", default=None, help="comma-separated layers to test (default: about a dozen, evenly "
                                                   "spread; see the docstring for BAGEL and Show-o2)")
    ap.add_argument("--concepts", default=None, help="comma-separated, default all of CONCEPTS")
    ap.add_argument("--sources", default=None, help="comma-separated first words of sources_for() labels, default all")
    ap.add_argument("--no-part-a", action="store_true")
    # SenseNova-U1 without its fixed generation system message (not used for the paper's numbers)
    ap.add_argument("--no-system", action="store_true")
    ap.add_argument("--folder", default=None, help="output folder name (default: check_<model name>)")
    args = ap.parse_args()
    concepts = args.concepts.split(",") if args.concepts else CONCEPTS
    sources = set(args.sources.split(",")) if args.sources else None
    m = load(args.model)
    if args.no_system:
        no_system_message(m)
    out = results.folder(args.folder or f"check_{m.name}")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else scan_layers(m.n_layers)
    numbers = dict(model=m.name, size=args.size, layers=layers, target=TARGET, patching={},
                   system_message=getattr(m, "system_message", None))
    if args.shard == 0 and not args.no_part_a:
        part_a(m, args.model, out, numbers, args.size)
        print({k: v for k, v in numbers.items() if k != "two_hop_answers"}, flush=True)

    judge = score.ClipJudge(prompts.concept_names())
    target = m.read_prompt(TARGET)
    slot = find_word(target, "thing")[-1]
    baseline = m.to_picture(m.draw(target, size=args.size, seed=0)[0])
    for concept in concepts[args.shard::args.of]:
        rows, labels = [], []
        for label, (text, word) in sources_for(concept).items():
            if sources is not None and label.split()[0].rstrip(":") not in sources:   # by first word: plain, one-hop, control
                continue
            src = m.read(text)
            pos = find_word(src, word)[-1] if word else int(torch.nonzero(src.content).flatten()[-1])
            pictures = [m.to_picture(m.draw(m.read_prompt(TARGET, edit=replace_at(l, [slot], src.hidden[l][pos][None])),
                                            size=args.size, seed=0)[0]) for l in layers]
            r = score.ranks(judge.similarities(pictures), [prompts.concept_index(concept)] * len(pictures))
            numbers["patching"].setdefault(label, {})[concept] = dict(source=text, token=src.token_strs[pos],
                                                                      ranks=[int(x) for x in r])
            rows.append(pictures)
            labels.append(f"{label} [{src.token_strs[pos].strip()}]")
            print(concept, label, src.token_strs[pos], [int(x) for x in r], flush=True)
        gallery.grid([[baseline] + row for row in rows], row_labels=labels,
                     col_labels=["no copy"] + [f"layer {l}" for l in layers], cell=112,
                     title=f"{m.name}, {concept}: hidden state copied into 'a photo of a thing' at one layer"
                     ).save(out / f"patching_{concept}.png")
    tag = f"_{args.concepts.replace(',', '-')}" if args.concepts else ""
    results.save_numbers(out / f"numbers_shard{args.shard}of{args.of}{tag}.json", numbers)
    print("->", out)


if __name__ == "__main__":
    main()
