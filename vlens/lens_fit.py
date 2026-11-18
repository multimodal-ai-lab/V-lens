"""A generic average-Jacobian fitter with running sums (used by scripts/fit_text_jlens.py for the text J-lens).

For one prompt the fitter computes, for every layer l,

    rows[l] = mean over source tokens p of  d y / d h_l[p]          shape [K, d_model]

where h_l[p] is the hidden state entering layer l at token p and y is a target computed from the reading (flattened),
optionally projected onto K directions. The derivative includes every path from h_l[p] to the target. One backward pass
per output number gives the rows of every layer at once, because the gradient is taken with respect to all recorded
hidden states together (they are nodes of one computation graph).

Averaging rows over many prompts gives the map. The fit keeps running sums and counts, so shards over disjoint prompts
merge exactly (sum of sums divided by sum of counts) and a stopped job resumes where it left off.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

from vlens.models.base import TextReading, UnifiedModel

# target(model, reading, prompt_index) -> flat float32 latent [D] that still carries the graph
Target = Callable[[UnifiedModel, TextReading, int], torch.Tensor]
# sources(reading) -> bool [n_tokens]: which tokens the rows are averaged over (default: the content tokens)
Sources = Callable[[TextReading], torch.Tensor]


def prompt_rows(model: UnifiedModel, text: str, target: Target, basis: Optional[torch.Tensor] = None,
                *, prompt_index: int = 0, layers: Optional[Sequence[int]] = None,
                sources: Optional[Sources] = None) -> dict:
    """The per-prompt Jacobian rows, plus the statistics needed to centre the lens later."""
    layers = list(range(model.n_layers)) if layers is None else list(layers)
    reading = model.read(text, keep_grad=True)
    out = target(model, reading, prompt_index)
    y = out if basis is None else basis.to(device=out.device, dtype=out.dtype) @ out
    hidden = [reading.hidden[l] for l in layers]
    mask = (reading.content if sources is None else sources(reading)).to(out.device)
    rows = torch.empty(len(layers), y.numel(), model.d_model, device=out.device, dtype=torch.float32)
    for k in range(y.numel()):
        grads = torch.autograd.grad(y[k], hidden, retain_graph=k < y.numel() - 1, allow_unused=True)
        rows[:, k] = torch.stack([
            torch.zeros(model.d_model, device=out.device) if g is None else g[mask].float().mean(0)
            for g in grads
        ])
    with torch.no_grad():
        hidden_sum = torch.stack([h[mask].float().sum(0) for h in hidden])
    return dict(rows=rows.cpu(), guess=out.detach().float().cpu(), hidden_sum=hidden_sum.cpu(),
                n_tokens=int(mask.sum()), layers=layers)


def empty_state(n_layers: int, n_directions: int, d_model: int, latent_size: int, meta: dict) -> dict:
    return dict(
        rows_sum=torch.zeros(n_layers, n_directions, d_model, dtype=torch.float32),
        guess_sum=torch.zeros(latent_size, dtype=torch.float32),
        hidden_sum=torch.zeros(n_layers, d_model, dtype=torch.float32),
        n_tokens=0, n_prompts=0, prompt_ids=[], meta=dict(meta),
    )


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fit(model: UnifiedModel, prompts: Sequence[str], target: Target, basis: Optional[torch.Tensor] = None, *,
        prompt_ids: Optional[Sequence[int]] = None, layers: Optional[Sequence[int]] = None,
        meta: Optional[dict] = None, checkpoint: Optional[str | Path] = None, checkpoint_every: int = 1,
        sources: Optional[Sources] = None, log: Callable[[str], None] = print) -> dict:
    """Accumulate prompt_rows over `prompts`. Returns the running-sum state (see `finish`).

    `prompt_ids` name the prompts (default 0..n-1); they are what shard merging checks for overlap.
    With `checkpoint`, the state is saved every `checkpoint_every` prompts (and at the end) and resumed from if
    the file exists. `sources` chooses the tokens the rows are averaged over (default: the content tokens).
    """
    prompt_ids = list(range(len(prompts))) if prompt_ids is None else list(prompt_ids)
    layers = list(range(model.n_layers)) if layers is None else list(layers)
    meta = dict(meta or {}, layers=layers)
    ckpt = Path(checkpoint) if checkpoint is not None else None
    state = None
    if ckpt is not None and ckpt.is_file():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if state["meta"] != meta:
            raise ValueError(f"{ckpt} was fitted with different settings; delete it to refit")
        log(f"resumed from {ckpt.name}: {state['n_prompts']} prompts already done")
    done = set(state["prompt_ids"]) if state else set()

    for text, pid in zip(prompts, prompt_ids):
        if pid in done:
            continue
        t0 = time.time()
        r = prompt_rows(model, text, target, basis, prompt_index=pid, layers=layers, sources=sources)
        if state is None:
            state = empty_state(len(layers), r["rows"].shape[1], model.d_model, r["guess"].numel(), meta)
        drift = float("nan")
        if state["n_prompts"]:
            old = state["rows_sum"] / state["n_prompts"]
            new = (state["rows_sum"] + r["rows"]) / (state["n_prompts"] + 1)
            drift = float((new - old).norm() / old.norm())
        state["rows_sum"] += r["rows"]
        state["guess_sum"] += r["guess"]
        state["hidden_sum"] += r["hidden_sum"]
        state["n_tokens"] += r["n_tokens"]
        state["n_prompts"] += 1
        state["prompt_ids"].append(pid)
        log(f"prompt {pid}: {r['n_tokens']} content tokens, {time.time() - t0:.1f}s, "
            f"change of the running average {drift:.2e}")
        if ckpt is not None and (state["n_prompts"] % checkpoint_every == 0):
            _save_atomic(state, ckpt)
    if ckpt is not None and state is not None:
        _save_atomic(state, ckpt)
    if state is None:
        raise ValueError("nothing was fitted")
    return state


def merge(states: Sequence[dict]) -> dict:
    """Combine states fitted on disjoint prompts. Exactly equal to one fit over all of them."""
    first = states[0]
    seen: set = set()
    for s in states:
        if s["meta"] != first["meta"]:
            raise ValueError("cannot merge fits made with different settings")
        overlap = seen & set(s["prompt_ids"])
        if overlap:
            raise ValueError(f"prompts fitted twice: {sorted(overlap)[:10]}")
        seen |= set(s["prompt_ids"])
    out = empty_state(*first["rows_sum"].shape, first["guess_sum"].numel(), first["meta"])
    for s in states:
        for key in ("rows_sum", "guess_sum", "hidden_sum"):
            out[key] += s[key]
        out["n_tokens"] += s["n_tokens"]
        out["n_prompts"] += s["n_prompts"]
        out["prompt_ids"] += list(s["prompt_ids"])
    return out


def finish(state: dict) -> dict:
    """Turn running sums into averages: the lens rows, the mean hidden state and the mean guess."""
    n = state["n_prompts"]
    return dict(
        rows=state["rows_sum"] / n,
        mean_hidden=state["hidden_sum"] / state["n_tokens"],
        mean_guess=state["guess_sum"] / n,
        n_prompts=n,
        prompt_ids=list(state["prompt_ids"]),
        meta=dict(state["meta"]),
    )


def cosine64(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine of two large tensors in float64 (float32 can return values above 1 on millions of entries)."""
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def agreement_by_layer(rows_a: torch.Tensor, rows_b: torch.Tensor) -> list[float]:
    """Per-layer cosine between two lenses fitted on different prompts: how stable the fit is."""
    return [cosine64(rows_a[l], rows_b[l]) for l in range(rows_a.shape[0])]
