"""Shared helpers of the depth, held-thought, two-state and state-question experiments: the V-lens applied to any
layer (with per-coordinate spreads for every layer), word positions, one-word answers, and batched drawing.

The V-lens here is the same transport as vlens/carry.py's "diag": v = mu_R + s_R D x / |D x| with x = h - mu_l and
D = diag(sigma_R / sigma_l), inserted into the slot "thing" of the model's drawing request at the insertion layer R.
"""

import re

import torch

from vlens.carry import CopyTransform, all_layer_sigma
from vlens.hidden import find_word, replace_at

REQUEST = {"sensenova": "thing"}  # every other model: "a photo of a thing"
ONE_WORD = " Answer with one word."
MAX_REPLY = 40


def layers(n_layers):
    """Every second layer (every third for models deeper than 30 layers), plus the last."""
    step = 3 if n_layers > 30 else 2
    return sorted(set(range(0, n_layers, step)) | {n_layers - 1})


def last_word(reading, word):
    """The last token of the last whole-word occurrence of `word` in the reading."""
    starts, full = [], ""
    for s in reading.token_strs:
        starts.append(len(full))
        full += s
    b = [h.start() for h in re.finditer(rf"(?<![\w-]){re.escape(word)}(?![\w-])", full)][-1]
    return [p for p, s in enumerate(starts) if s < b + len(word) and s + len(reading.token_strs[p]) > b][-1]


def names(it):
    return it.get("bridge_names", [it["truth"]])


def answer_hit(reply, answers):
    """The first whole-word occurrence of a right answer in the reply: (start, end) or None."""
    hits = [h.span() for a in answers for h in re.finditer(rf"(?<![\w-]){re.escape(a)}(?![\w-])", reply.lower())]
    return min(hits) if hits else None


def all_sigma(m, s, name):
    """Per-coordinate spreads at every layer: the stored ones when every layer was stored (HunyuanImage, Liquid,
    Janus-Pro), else computed once from the passages (vlens.carry.all_layer_sigma)."""
    if len(s["sigma"]) == m.n_layers:
        return torch.stack([s["sigma"][l].float() for l in range(m.n_layers)])
    return all_layer_sigma(m, s, name).float()


class Lens:
    """The V-lens from any layer of a loaded model `m` (statistics LENS_STORE/<name>_copy_stats)."""

    def __init__(self, m, name, request=None):
        self.m, self.s = m, CopyTransform(name).s
        self.R, self.mu = self.s["R"], self.s["mean"].float()
        self.sigma = all_sigma(m, self.s, name)
        self.request = request or REQUEST.get(name, "a photo of a thing")
        self.slot = find_word(m.read_prompt(self.request), "thing")[-1]

    def reading(self, h, layer):
        z = (h.float().cpu() - self.mu[layer]) * (self.sigma[self.R] / self.sigma[layer])
        v = self.mu[self.R] + z * (self.s["spread"] / z.norm())
        return self.m.read_prompt(self.request, edit=replace_at(self.R, [self.slot], v[None].to(self.m.device)))


def draw_all(m, jobs, out, batch, limit=None):
    """jobs: (file, reading, seed). Pictures that exist are skipped; with draw_many, readings of one seed and one
    length go together (SenseNova's draw_many needs equal lengths; the lens readings all share the request's)."""
    todo = [j for j in jobs if not (out / j[0]).exists()][:limit]
    many = getattr(m, "draw_many", None) if batch > 1 else None
    for seed, n in sorted({(j[2], len(j[1].token_ids)) for j in todo}):
        group = [j for j in todo if j[2] == seed and len(j[1].token_ids) == n]
        if many is None:
            for f, reading, _ in group:
                m.to_picture(m.draw(reading, seed=seed)[0]).save(out / f)
            continue
        for k in range(0, len(group), batch):
            chunk = group[k:k + batch]
            x = many([r for _, r, _ in chunk], seed=seed)
            for i, (f, _, _) in enumerate(chunk):
                m.to_picture(x[i:i + 1]).save(out / f)
