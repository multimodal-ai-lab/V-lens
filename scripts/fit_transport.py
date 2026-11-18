"""Fit the layer statistics and the Jacobian maps that carry a hidden state from any layer to the insertion layer R.

The layer averages mu_l (64 fitting passages) and the radius s_R (median distance from mu_R on 20 held-out passages)
are part of the V-lens (paper Eq. 1, vlens/carry.py). The average Jacobian maps are the "Jacobian map" lens variant of
Table 1 and Supp. Table S2 (and the base of the "Jacobian map with gain" variant, scripts/self_gain.py).

Three steps:
1. fit (one GPU per shard): same-token average Jacobians on plain text passages (configs/prompts/text_lens_passages.txt,
   passages 0..63; passage i goes to shard i mod n): forward from every layer below R to R, and backward from R to the
   later layers in BACKWARD (inverted when used, see vlens/transport.py). The shards also sum the hidden states for
   the layer averages;
2. merge (CPU): average the shards into one Transport (LENS_STORE/<model>_transport_to<R>_n64/transport.pt);
3. validate (one GPU): on 20 held-out passages (280..299), carry each token's hidden state from layer l to R and
   compare with its real state at R (baselines: the unchanged copy and the average state at R). This step stores the
   radius s_R ("spread") and the typical distance at every layer, and picks the ridge strength of the backward maps.

The model is chosen with --model (bagel, show_o2 or sensenova; R in vlens/models/__init__.py).

    python scripts/fit_transport.py --model bagel --shard 0 --of 32      # one call per shard, 0..31
    python scripts/fit_transport.py --model bagel --of 32 --merge
    python scripts/fit_transport.py --model bagel --validate
"""

import argparse
from pathlib import Path

import torch

from vlens import paths, prompts, results
from vlens.models import READING_LAYER, load, transport_name
from vlens.transport import Transport, band, passage_rows

DATA = paths.DATA
# later layers whose backward maps are fitted (and inverted when used), per model
BACKWARD = {"bagel": [10, 12, 14, 16, 20, 24], "show_o2": [], "sensenova": [28, 32, 36, 40]}
FIT_IDS = list(range(64))  # replaced by range(--passages) in main(); the held-out passages start at 280
HELD_OUT_IDS = list(range(280, 300))
RIDGES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, float("inf")]  # inf: the transpose of the average Jacobian


def passages():
    return (prompts.PROMPTS / "text_lens_passages.txt").read_text().splitlines()


def name_of(args):
    return transport_name(args.model, len(FIT_IDS))


def backward_targets(model, R):
    return [l for l in BACKWARD[model] if l > R]


def fit_shard(args):
    import time

    m = load(args.model)
    R = READING_LAYER[args.model]
    pool = passages()
    folder = DATA / "fits" / name_of(args)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"shard{args.shard:03d}of{args.of:03d}.pt"
    state = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None
    for pid in FIT_IDS[args.shard::args.of]:
        if state and pid in state["ids"]:
            continue
        t0 = time.time()
        r = passage_rows(m, pool[pid], R, backward_targets(args.model, R), seed=pid)
        if state is None:
            state = dict(forward=torch.zeros_like(r["forward"]), backward={l: torch.zeros_like(b) for l, b in r["backward"].items()},
                         hidden_sum=torch.zeros_like(r["hidden_sum"]), n_tokens=0, ids=[])
        state["forward"] += r["forward"]
        for l, b in r["backward"].items():
            state["backward"][l] += b
        state["hidden_sum"] += r["hidden_sum"]
        state["n_tokens"] += r["n_tokens"]
        state["ids"].append(pid)
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        print(f"passage {pid}: {r['n_tokens']} tokens, {time.time() - t0:.0f}s", flush=True)
    print(f"shard {args.shard}/{args.of} done")


def merge(args):
    name, R = name_of(args), READING_LAYER[args.model]
    folder = DATA / "fits" / name
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(folder.glob(f"shard*of{args.of:03d}.pt"))]
    ids = sorted(i for s in states for i in s["ids"])
    if ids != FIT_IDS:
        raise SystemExit(f"shards cover {len(ids)} passages, expected {len(FIT_IDS)}")
    n = len(ids)
    forward = sum(s["forward"] for s in states) / n
    backward = {l: sum(s["backward"][l] for s in states) / n for l in backward_targets(args.model, R)}
    mean = sum(s["hidden_sum"] for s in states) / sum(s["n_tokens"] for s in states)
    t = Transport(R, forward, backward, mean,
                  meta=dict(passages="configs/prompts/text_lens_passages.txt", ids=ids, rules="plain",
                            estimator="same-token average Jacobian, random signs"))
    folder_out = t.save(name)
    print("saved", folder_out)


@torch.no_grad()
def validate(args):
    name = name_of(args)
    t = Transport.load(name)
    m = load(args.model)
    pool = passages()
    H = {l: [] for l in range(m.n_layers)}
    for pid in HELD_OUT_IDS:
        r = m.read(pool[pid])
        mask = band(r.n_tokens).to(r.hidden[0].device)
        for l in range(m.n_layers):
            H[l].append(r.hidden[l][mask].float().cpu())
    H = {l: torch.cat(v) for l, v in H.items()}
    target = H[t.R]
    spread = (target - t.mean[t.R]).norm(dim=1)
    t.meta["spread"] = float(spread.median())  # typical distance of a real reading-layer state from the average
    t.meta["spread_by_layer"] = {l: float((H[l] - t.mean[l]).norm(dim=1).median()) for l in H}  # for size="relative"

    def error(v):
        return float(((v - target).norm(dim=1) / spread).mean())

    def cosine(v):
        a, b = v - t.mean[t.R], target - t.mean[t.R]
        return float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean())

    numbers = dict(reading_layer=t.R, held_out=HELD_OUT_IDS, n_tokens=int(target.shape[0]), by_layer={})
    for l in t.layers:
        if l == t.R:
            continue
        copy = H[l]
        entry = dict(copy=dict(error=error(copy), cosine=cosine(copy)),
                     average=dict(error=error(t.mean[t.R].expand_as(target))))
        if l < t.R:
            v = t.to_reading_layer(H[l], l)
            entry["jacobian"] = dict(error=error(v), cosine=cosine(v))
        else:
            # the ridge is chosen by direction (cosine): the size is set separately by matching (see Transport)
            tries = {frac: t.to_reading_layer(H[l], l, frac) for frac in RIDGES}
            coss = {frac: cosine(v) for frac, v in tries.items()}
            best = max(coss, key=coss.get)
            t.ridge[l] = best
            entry["jacobian"] = dict(error=error(tries[best]), cosine=coss[best], ridge=best, all_ridges=coss)
        entry["jacobian_size_matched"] = dict(error=error(t.to_reading_layer(H[l], l, match_size=True)))
        copied = H[l] - t.mean[l]
        copied = t.mean[t.R] + copied * (t.meta["spread"] / copied.norm(dim=-1, keepdim=True))
        entry["copy_size_matched"] = dict(error=error(copied))
        numbers["by_layer"][l] = entry
        print(f"layer {l:2d} -> {t.R}: error  jacobian {entry['jacobian']['error']:.3f}  size-matched {entry['jacobian_size_matched']['error']:.3f}  "
              f"copy {entry['copy']['error']:.3f}  copy size-matched {entry['copy_size_matched']['error']:.3f}  average {entry['average']['error']:.3f}"
              f"   cosine jacobian {entry['jacobian']['cosine']:.3f}  copy {entry['copy']['cosine']:.3f}")
    t.save(name)
    out = results.folder(f"fit_{name}")
    results.save_numbers(out / "numbers.json", numbers)
    print("->", out)


def main():
    ap = argparse.ArgumentParser(description="Fit, merge and validate the layer statistics and Jacobian maps to R.")
    ap.add_argument("--model", default="bagel")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=32)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--passages", type=int, default=64, help="fitting passages 0..n-1 (at most 280; the paper: 64)")
    args = ap.parse_args()
    global FIT_IDS
    assert args.passages <= HELD_OUT_IDS[0]
    FIT_IDS = list(range(args.passages))
    if args.merge:
        merge(args)
    elif args.validate:
        validate(args)
    else:
        fit_shard(args)


if __name__ == "__main__":
    main()
