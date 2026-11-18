"""Compute the V-lens statistics (paper Eq. 1) of BAGEL, Show-o2 or SenseNova-U1 on the CPU.

Needs the transport of scripts/fit_transport.py (layer averages mu_l over the 64 fitting passages and the radius s_R from
the 20 held-out passages, after --validate) and the saved states of scripts/collect_states.py (--splits
fit,heldout,calib). The per-coordinate spreads sigma_l come from the fitting and the 40 further passages, their squares
floored at 1% of the layer's median coordinate variance. Writes LENS_STORE/<model>_copy_stats/stats.pt, which
vlens.carry.CopyTransform and vlens.readout.Lens read. (HunyuanImage, Liquid and Janus-Pro: scripts/lens_stats_own_tokens.py.)

    python scripts/build_lens_stats.py --model bagel
"""

import argparse

import torch

from vlens import carry


def main():
    ap = argparse.ArgumentParser(description="Compute the V-lens statistics from the transport and the saved states.")
    ap.add_argument("--model", required=True, choices=["bagel", "show_o2", "sensenova"])
    args = ap.parse_args()
    folder = carry.build(args.model)
    s = torch.load(folder / "stats.pt", map_location="cpu", weights_only=False)
    print(f"{args.model}: R = {s['R']}, s_R = {s['spread']:.3f}, layers with spreads {s['layers']}")
    print("->", folder / "stats.pt")


if __name__ == "__main__":
    main()
