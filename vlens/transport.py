"""Carry a hidden state from one layer to another at the same token, with an average Jacobian.

These maps give the "Jacobian map" variants of the lens (Table 1): a hidden state from layer l is carried to the
insertion (reading) layer R before it is placed into the slot of the drawing request. The layer averages and the typical
distance from them, stored with the maps, are also the means mu_l and the radius s_R of the V-lens (vlens/carry.py).

- l < R, forward: the average Jacobian F_l = average over passages and tokens p of d h_R[p] / d h_l[p];
- l > R, backward: the average Jacobian B_l = average of d h_l[p] / d h_R[p], inverted by ridge-regularized least
  squares (a forward Jacobian points the wrong way).

Both are used as affine maps around the average hidden states, so the average state maps to the average state:
    forward   v = mean_R + F_l (h - mean_l)
    backward  v = mean_R + (B_l^T B_l + ridge I)^-1 B_l^T (h - mean_l)

Same-token derivatives from one backward pass per output number: the cotangent puts a random +1 or -1 on each token
and the gradient at token p is multiplied by the same sign. Terms from other tokens then cancel on average and the
same-token derivative remains (exact when only one token is used).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch

from vlens.paths import LENS_STORE

SKIP_FIRST = 4


def band(n_tokens: int, skip_first: int = SKIP_FIRST) -> torch.Tensor:
    """Tokens used for fitting: from the 5th up to the last before the end marker."""
    mask = torch.zeros(n_tokens, dtype=torch.bool)
    mask[skip_first:n_tokens - 1] = True
    return mask


def _rows(y: torch.Tensor, sources: Sequence[torch.Tensor], signs: torch.Tensor, n_band: int) -> torch.Tensor:
    """For every output number i of y: the signed average over tokens of d y_i / d source. [n_sources, d_out, d_in]."""
    d = y.numel()
    out = torch.empty(len(sources), d, sources[0].shape[-1], device=y.device)
    for i in range(d):
        grads = torch.autograd.grad(y[i], list(sources), retain_graph=True)
        signed = [signs[:, None] * g.float() for g in grads]
        out[:, i] = torch.stack([w.sum(0) for w in signed]) / n_band
    return out


def passage_rows(model, text: str, reading_layer: int, backward_targets: Sequence[int], seed: int,
                 mask: Optional[torch.Tensor] = None) -> dict:
    """Same-token Jacobian rows for one passage: forward (layers below the reading layer) and backward targets, plus
    the sum of the hidden states over the passage's band tokens at every layer (for the layer averages)."""
    reading = model.read(text, keep_grad=True)
    mask = band(reading.n_tokens) if mask is None else mask
    g = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (reading.n_tokens,), generator=g) * 2 - 1).float()
    signs[~mask] = 0.0
    device = reading.hidden[0].device
    s = signs.to(device)
    n_band = int(mask.sum())
    R = reading_layer

    y = (s[:, None] * reading.hidden[R].float()).sum(0)
    forward = None
    if R > 0:
        forward = _rows(y, [reading.hidden[l] for l in range(R)], s, n_band).cpu()
    backward = {}
    for l in backward_targets:
        y = (s[:, None] * reading.hidden[l].float()).sum(0)
        backward[l] = _rows(y, [reading.hidden[R]], s, n_band)[0].cpu()
    with torch.no_grad():
        m = mask.to(device)
        hidden_sum = torch.stack([h[m].float().sum(0) for h in reading.hidden]).cpu()
    return dict(forward=forward, backward=backward, hidden_sum=hidden_sum, n_tokens=n_band)


class Transport:
    """Fitted transport maps to one reading layer. Built by scripts/fit_transport.py."""

    def __init__(self, reading_layer: int, forward: Optional[torch.Tensor], backward: dict, mean: torch.Tensor,
                 ridge: Optional[dict] = None, meta: Optional[dict] = None, direct: Optional[dict] = None):
        self.R = reading_layer
        self.direct = dict(direct or {})  # {layer: [d, d]} explicit maps (the self-gain variant); used first if present
        self.forward = forward  # [R, d, d]
        self.backward = backward  # {layer: [d, d]}
        self.mean = mean  # [n_layers, d]
        self.ridge = dict(ridge or {})  # {layer: ridge strength as a fraction of the average eigenvalue}
        self.meta = dict(meta or {})
        self._solvers: dict = {}

    @property
    def layers(self) -> list[int]:
        return sorted(list(range(self.R + 1)) + list(self.backward))

    def solver(self, layer: int, ridge_fraction: float) -> torch.Tensor:
        """(B^T B + ridge I)^-1 B^T. An infinite ridge gives the limit direction, B^T itself (the size is set
        separately when matching sizes)."""
        key = (layer, ridge_fraction)
        if key not in self._solvers and ridge_fraction == float("inf"):
            self._solvers[key] = self.backward[layer].T.float().contiguous()
        if key not in self._solvers:
            B = self.backward[layer].double()
            gram = B.T @ B
            lam = ridge_fraction * float(torch.diagonal(gram).mean())
            self._solvers[key] = torch.linalg.solve(gram + lam * torch.eye(gram.shape[0], dtype=gram.dtype), B.T).float()
        return self._solvers[key]

    def to_reading_layer(self, h: torch.Tensor, layer: int, ridge_fraction: Optional[float] = None,
                         match_size: bool = False, size: str = "fixed") -> torch.Tensor:
        """Carry h ([d] or [n, d]) from `layer` to the reading layer.

        With match_size, the carried state's distance from the average state is set by `size`:
        - "fixed": the typical distance of a real reading-layer state (`self.meta["spread"]`, measured on held-out
          text). The backward maps get the direction partly right but the size badly wrong;
        - "relative": that typical distance times h's own distance from its layer's average relative to the typical
          distance at its layer (`self.meta["spread_by_layer"]`), so a state that stands out at its layer stands out
          as much at the reading layer;
        - "none": as the map gives it (the same as match_size=False).
        """
        h = h.float().cpu()
        if layer in self.direct:
            v = self.mean[self.R] + (h - self.mean[layer]) @ self.direct[layer].float().T
        elif layer == self.R:
            v = h
        elif layer < self.R:
            v = self.mean[self.R] + (h - self.mean[layer]) @ self.forward[layer].T
        elif layer in self.backward:
            frac = self.ridge.get(layer, 1e-3) if ridge_fraction is None else ridge_fraction
            v = self.mean[self.R] + (h - self.mean[layer]) @ self.solver(layer, frac).T
        else:
            raise ValueError(f"no backward map fitted for layer {layer}; fitted: {sorted(self.backward)}")
        if match_size and size != "none":
            dev = v - self.mean[self.R]
            target = torch.as_tensor(self.meta["spread"], dtype=dev.dtype)
            if size == "relative":
                by_layer = self.meta["spread_by_layer"]
                own = (h - self.mean[layer]).norm(dim=-1, keepdim=True)
                target = target * own / float(by_layer[layer] if layer in by_layer else by_layer[str(layer)])
            elif size != "fixed":
                raise ValueError(f"unknown size {size!r}; known: fixed, relative, none")
            v = self.mean[self.R] + dev * (target / dev.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        return v

    def save(self, name: str, store: Path = LENS_STORE) -> Path:
        folder = Path(store) / name
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(dict(reading_layer=self.R, forward=self.forward, backward=self.backward, mean=self.mean,
                        ridge=self.ridge, meta=self.meta, direct=self.direct), folder / "transport.pt")
        return folder

    @classmethod
    def load(cls, name: str, store: Path = LENS_STORE) -> "Transport":
        blob = torch.load(Path(store) / name / "transport.pt", map_location="cpu", weights_only=False)
        return cls(blob["reading_layer"], blob["forward"], blob["backward"], blob["mean"], blob["ridge"], blob["meta"],
                   blob.get("direct"))
