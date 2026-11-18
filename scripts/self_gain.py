"""The "Jacobian map with gain" lens variant (Table 1, Supp. Table S2): the average Jacobian plus one scalar per layer
for each state's growth along its own direction (CPU or GPU, no model needed).

Between layers, a token's state grows along its own direction, but the average Jacobian's correction does not express
a gain that depends on the state itself. This variant adds it back with one scalar per layer:
    A_l = M_l + gamma_l I,   gamma_l = <y - M_l x, x> / <x, x>  over all calibration tokens,
where M_l is the standard map (forward average Jacobian, or the ridge inverse of the backward one; scripts/fit_transport.py),
x = h_l - mu_l and y = h_R - mu_R on the calibration passages 240 to 279 (scripts/collect_states.py --splits calib).
No picture is used. Reported on the held-out passages 280 to 299: cosine with the true insertion-layer state for the
standard map, the moved copy and the gain map, and the own-state coefficient <v, x>/<x, x> of each against the true one.
--save writes the gain maps as a transport `<standard>_selfgain` (explicit maps, used first by
Transport.to_reading_layer), drawn by scripts/draw_cells.py --ways vlens --transport <model>_transport_to<R>_n64_selfgain.

    python scripts/self_gain.py --model bagel --save
"""

import argparse

import torch

from vlens import paths, results
from vlens.models import transport_name
from vlens.transport import Transport

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def states(model, layer, split):
    return torch.load(paths.DATA / "states" / model / f"{split}_L{layer}.pt").float().to(DEVICE)


def main():
    ap = argparse.ArgumentParser(description="Fit the Jacobian-plus-gain maps from the saved passage states.")
    ap.add_argument("--model", default="bagel")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(32)
    std = Transport.load(transport_name(args.model))
    R = std.R
    mu = std.mean.float().to(DEVICE)
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean())  # noqa: E731
    own = lambda v, x: float(((v * x).sum(1) / (x * x).sum(1)).mean())  # noqa: E731
    y_cal, y_held = states(args.model, R, "calib") - mu[R], states(args.model, R, "heldout") - mu[R]
    numbers, direct = dict(model=args.model, by_layer={}), {}
    saved = sorted(int(f.stem.split("_L")[1]) for f in (paths.DATA / "states" / args.model).glob("calib_L*.pt"))
    for layer in [l for l in saved if l < R or l in std.backward]:  # layers with saved states and a map
        backward = layer > R
        M = (std.solver(layer, std.ridge.get(layer, 1e-3)) if backward else std.forward[layer]).float().to(DEVICE)
        x_cal, x_held = states(args.model, layer, "calib") - mu[layer], states(args.model, layer, "heldout") - mu[layer]
        r = y_cal - x_cal @ M.T
        gamma = float((r * x_cal).sum() / (x_cal * x_cal).sum())
        A = M + gamma * torch.eye(M.shape[0], device=DEVICE)
        v_std, v_new = x_held @ M.T, x_held @ A.T
        entry = dict(gamma=gamma, cos_standard=cos(v_std, y_held), cos_moved=cos(x_held, y_held),
                     cos_selfgain=cos(v_new, y_held), own_standard=own(v_std, x_held), own_selfgain=own(v_new, x_held),
                     own_true=own(y_held, x_held))
        numbers["by_layer"][layer] = entry
        direct[layer] = A.cpu()
        print(f"layer {layer:2d}: gamma {gamma:+.3f}  cos with true: standard {entry['cos_standard']:.3f}  moved "
              f"{entry['cos_moved']:.3f}  self-gain {entry['cos_selfgain']:.3f}  |  own-state coefficient: standard "
              f"{entry['own_standard']:+.2f}  self-gain {entry['own_selfgain']:+.2f}  true {entry['own_true']:+.2f}", flush=True)
    out = results.folder("self_gain")
    results.save_numbers(out / f"numbers_{args.model}.json", numbers)
    if args.save:
        t = Transport(R, std.forward, std.backward, std.mean, std.ridge,
                      dict(std.meta, variant="self-gain: average Jacobian plus one scalar per layer"), direct=direct)
        print("saved", t.save(transport_name(args.model) + "_selfgain"))


if __name__ == "__main__":
    main()
