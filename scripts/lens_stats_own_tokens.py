"""The V-lens statistics of HunyuanImage-3.0, Liquid and Janus-Pro (--model), computed straight from the passages.

- Insertion layer R: read from the calibration (scripts/calibrate_insertion_layer.py, folder check_<model name>,
  plain-sentence ranks): the latest tested layer with the most of the 12 calibration words ranked first by CLIP
  (Supp. Table S4: HunyuanImage 9; Supp. I: Liquid 6, Janus-Pro 8).
- Statistics from the passages of configs/prompts/text_lens_passages.txt, read as the lens reads a text (m.read, a user
  message): means over passages 0 to 63, coordinate spreads and distances over passages 0 to 63 and 240 to 279, the
  squared spreads floored at 1% of the layer's median; s_R is the median distance from mu_R.
- One difference from the other models: HunyuanImage's text template carries a 1,222-token system prompt, so a token
  band from the 5th token to the end would be almost all system prompt. Only the passage's own tokens are used, for all
  three models.

Writes LENS_STORE/<model>_copy_stats/stats.pt (every layer), which vlens.carry.CopyTransform and vlens.readout.Lens read.

    python scripts/lens_stats_own_tokens.py --model hunyuan      # four 80 GB GPUs
    python scripts/lens_stats_own_tokens.py --model liquid
    python scripts/lens_stats_own_tokens.py --model janus_pro
"""

import argparse
import json

import torch

from vlens import prompts, results
from vlens.carry import FLOOR
from vlens.models import load
from vlens.paths import LENS_STORE

RESULTS = results.RESULTS
# the calibration folder of each model (scripts/calibrate_insertion_layer.py); Liquid's text template is short, but the
# same own-tokens rule keeps the models comparable
CHECK = {"hunyuan": RESULTS / "check_HunyuanImage-3.0", "liquid": RESULTS / "check_Liquid-7B",
         "janus_pro": RESULTS / "check_Janus-Pro-7B"}


def insertion_layer(check):
    ranks, layers = {}, None
    for f in check.glob("numbers_*.json"):
        n = json.loads(f.read_text())
        layers = n["layers"]
        ranks.update({c: v["ranks"] for c, v in n["patching"].get("plain sentence", {}).items()})
    firsts = [sum(r[i] == 1 for r in ranks.values()) for i in range(len(layers))]
    best = max(firsts)
    return max(l for l, k in zip(layers, firsts) if k == best), dict(zip(layers, firsts)), len(ranks)


def main():
    ap = argparse.ArgumentParser(description="V-lens statistics of HunyuanImage-3.0, Liquid or Janus-Pro.")
    ap.add_argument("--model", default="hunyuan", choices=sorted(CHECK))
    args = ap.parse_args()
    STATS = LENS_STORE / f"{args.model}_copy_stats" / "stats.pt"
    R, firsts, n = insertion_layer(CHECK[args.model])
    print(f"insertion layer {R} (CLIP first choice by layer over {n} concepts: {firsts})", flush=True)
    m = load(args.model)
    pool = (prompts.PROMPTS / "text_lens_passages.txt").read_text().splitlines()
    L = m.n_layers

    def states(ids):
        for pid in ids:
            r = m.read(pool[pid])
            yield torch.stack([h[r.content].float() for h in r.hidden])  # [L, n, d], the passage's own tokens

    s1, n1 = 0.0, 0
    for x in states(range(64)):
        s1 = s1 + x.sum(1)
        n1 += x.shape[1]
    mean = s1 / n1
    ss, n2, dists = 0.0, 0, [[] for _ in range(L)]
    for x in states(list(range(64)) + list(range(240, 280))):
        d = x - mean[:, None]
        ss = ss + d.pow(2).sum(1)
        n2 += x.shape[1]
        for l in range(L):
            dists[l].append(d[l].norm(dim=1))
    var = ss / n2
    sigma = {l: var[l].clamp(min=FLOOR * float(var[l].median())).sqrt() for l in range(L)}
    radii = {l: torch.cat(dists[l]).sort().values for l in range(L)}
    spread_by_layer = {l: float(radii[l].median()) for l in range(L)}
    STATS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(mean=mean, spread=spread_by_layer[R], R=R, spread_by_layer=spread_by_layer, sigma=sigma, radii=radii,
                    layers=list(range(L)), firsts=firsts, tokens=n2, content_only=True), STATS)
    print("->", STATS, n1, "mean tokens,", n2, "spread tokens;", {l: round(v, 1) for l, v in spread_by_layer.items()},
          flush=True)


if __name__ == "__main__":
    main()
