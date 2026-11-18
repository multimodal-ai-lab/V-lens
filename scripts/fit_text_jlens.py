"""Fit a text J-lens for a model's text side (the text readouts of Supp. D).

The text J-lens reads a hidden state out as words. It uses the J-lens estimator: the average derivative of the sum of
final states over later positions. For the hidden state entering layer l it averages, over passages and over source
tokens p, how the output of the last layer (before the final norm), summed over p and every later token p', changes
with the state at p (the sum over p' >= p follows from the causal attention):

    J[l][i, :] = average over p of  d ( sum over p' of h_last[p'][i] ) / d h_l[p]

A hidden state h at layer l is read out as words by lm_head(final_norm(J[l] h)) (vlens/text_lens.py). Here the state
is the one entering layer l, and the target is the last layer's output.

Source and target tokens: from the 5th token on (the first four act as attention sinks). For BAGEL, up to the last
token before its end marker; for Show-o2 and SenseNova-U1, the text's own tokens (the reading's content mask) from the
5th one on, which leaves out their start markers and chat prefix. One backward pass per hidden dimension, so about 1.5
to 2 minutes per passage on one GPU. Passages: the first 200 of configs/prompts/text_lens_passages.txt, unrelated to
the test items. HunyuanImage-3.0 has no J-lens (one backward pass per hidden dimension through 80B parameters).

    # one job per shard (passage i goes to shard i mod n), then merge on the CPU
    python scripts/fit_text_jlens.py --model bagel --n-passages 200 --shard 3 --of 25
    python scripts/fit_text_jlens.py --model bagel --n-passages 200 --of 25 --merge

Writes $VLENS_DATA/lenses/<model>_text_jlens_n200/text_lens.pt (shards in $VLENS_DATA/fits/).
"""

import argparse

import torch

from vlens import lens_fit, paths, prompts, results
from vlens.paths import LENS_STORE

DATA = paths.DATA
SKIP_FIRST = 4


def band(reading) -> torch.Tensor:
    """Source and target tokens: from the 5th token up to the last one before the end marker."""
    mask = torch.zeros(reading.n_tokens, dtype=torch.bool)
    mask[SKIP_FIRST:reading.n_tokens - 1] = True
    return mask


def content_band(reading) -> torch.Tensor:
    """For Show-o2 and SenseNova: the text's own tokens from the 5th one on."""
    mask = reading.content.clone()
    for p in [i for i in range(reading.n_tokens) if mask[i]][:SKIP_FIRST]:
        mask[p] = False
    return mask


def target(model, reading, prompt_index):
    b = band(reading) if model.name.lower().startswith("bagel") else content_band(reading)
    return reading.last_output[b.to(reading.last_output.device)].float().sum(0)


def name(args) -> str:
    return f"{args.model}_text_jlens_n{args.n_passages}"


def passages():
    return (prompts.PROMPTS / "text_lens_passages.txt").read_text().splitlines()


def fit_shard(args):
    from vlens.models import load

    m = load(args.model)
    pool = passages()
    ids = list(range(args.n_passages))[args.shard::args.of]
    folder = DATA / "fits" / name(args)
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(model=m.name, kind="text J-lens", target="output of the last layer, summed over later tokens",
                skip_first=SKIP_FIRST)
    lens_fit.fit(m, [pool[i] for i in ids], target, None, prompt_ids=ids, meta=meta,
                 sources=band if args.model == "bagel" else content_band,
                 checkpoint=folder / f"shard{args.shard:03d}of{args.of:03d}.pt", checkpoint_every=4)
    print(f"shard {args.shard}/{args.of} done: {len(ids)} passages")


def merge(args):
    folder = DATA / "fits" / name(args)
    files = sorted(folder.glob(f"shard*of{args.of:03d}.pt"))
    if len(files) != args.of:
        raise SystemExit(f"found {len(files)} of {args.of} shards in {folder}")
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
    got = sum(s["n_prompts"] for s in states)
    if got != args.n_passages:
        raise SystemExit(f"shards hold {got} passages, expected {args.n_passages}")
    whole = lens_fit.finish(lens_fit.merge(states))
    half_a = lens_fit.finish(lens_fit.merge(states[0::2]))
    half_b = lens_fit.finish(lens_fit.merge(states[1::2]))
    agreement = lens_fit.agreement_by_layer(half_a["rows"], half_b["rows"])
    out = LENS_STORE / name(args)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(J=whole["rows"].half(), mean_hidden=whole["mean_hidden"], n_passages=whole["n_prompts"],
                    meta=dict(whole["meta"], half_agreement_by_layer=agreement)), out / "text_lens.pt")
    res = results.folder(f"fit_{name(args)}")
    results.save_numbers(res / "numbers.json", dict(n_passages=whole["n_prompts"], half_agreement_by_layer=agreement))
    for l, a in enumerate(agreement):
        print(f"layer {l:2d}: agreement between halves {a:.3f}")
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="bagel")
    ap.add_argument("--n-passages", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    merge(args) if args.merge else fit_shard(args)


if __name__ == "__main__":
    main()
