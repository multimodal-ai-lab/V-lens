"""Save the hidden states of the statistics passages, from which the V-lens statistics are computed.

For one model: read passages of configs/prompts/text_lens_passages.txt and keep, for the tokens of the transport's band
(from the 5th token to the token before the final marker, vlens.transport.band), the hidden state entering each source
layer and the insertion layer. Splits: fit (passages 0..63), calib (240..279) and heldout (280..299). The V-lens
statistics (vlens/carry.py, scripts/build_lens_stats.py) need --splits fit,heldout,calib; the spreads sigma use the
fit and calib splits. --all-layers keeps every layer instead of the source layers below. Saved as float16 under
$VLENS_DATA/states/<model>/<split>_L<l>.pt, each [n_tokens, d], tokens in the same order at every layer.

    python scripts/collect_states.py --model bagel --splits fit,heldout,calib
"""

import argparse

import torch

from vlens import paths, prompts
from vlens.models import READING_LAYER, load
from vlens.transport import band

# the source layers kept without --all-layers (every layer the Jacobian maps cover)
SOURCE_LAYERS = {"bagel": [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24], "show_o2": [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24],
                 "sensenova": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]}
SPLITS = {"fit": list(range(64)), "heldout": list(range(280, 300)),
          "calib": list(range(240, 280))}  # calib: the 40 further passages of the coordinate spreads


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Save the hidden states of the statistics passages.")
    ap.add_argument("--model", default="bagel")
    ap.add_argument("--splits", default="fit,heldout")
    ap.add_argument("--all-layers", action="store_true", help="every layer, not only the source layers")
    args = ap.parse_args()
    out = paths.DATA / "states" / args.model
    out.mkdir(parents=True, exist_ok=True)
    m = load(args.model)
    layers = (list(range(m.n_layers)) if args.all_layers
              else sorted(set(SOURCE_LAYERS[args.model]) | {READING_LAYER[args.model]}))
    pool = (prompts.PROMPTS / "text_lens_passages.txt").read_text().splitlines()
    for name in args.splits.split(","):
        ids = SPLITS[name]
        kept = {l: [] for l in layers}
        for pid in ids:
            r = m.read(pool[pid])
            mask = band(r.n_tokens).to(r.hidden[0].device)
            for l in layers:
                kept[l].append(r.hidden[l][mask].to(torch.float16).cpu())
        for l in layers:
            torch.save(torch.cat(kept[l]), out / f"{name}_L{l}.pt")
        print(name, {l: tuple(torch.cat(kept[l]).shape) for l in layers[:1]}, "layers", layers, flush=True)


if __name__ == "__main__":
    main()
