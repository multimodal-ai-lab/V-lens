"""Reading hidden states out as words: the text J-lens and the plain logit lens (the text readouts of Supp. D).

- Logit lens: final norm + text head applied to the hidden state as it is.
- Text J-lens (fitted by scripts/fit_text_jlens.py): the hidden state is first carried to the last layer with
  the average Jacobian, then final norm + text head.

A concept counts as found at rank r if the best of its spellings (" elephant", "elephant", " Elephant", ...; the
first token of each) is the r-th highest score among all text tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from vlens.paths import LENS_STORE


class TextLens:
    def __init__(self, name: str = "bagel_text_jlens_n200", device: str = "cuda"):
        blob = torch.load(Path(LENS_STORE) / name / "text_lens.pt", map_location="cpu", weights_only=False)
        self.J = blob["J"].to(device)  # [L, d, d] fp16
        self.meta = blob["meta"]
        self.device = device

    def transport(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        return h.to(self.device, torch.float16) @ self.J[layer].T


def word_ids(tokenizer, word: str) -> list[int]:
    """First token of each common spelling of `word`."""
    ids = set()
    for variant in (f" {word}", word, f" {word.capitalize()}", word.capitalize(), f" {word.lower()}"):
        tokens = tokenizer.encode(variant)
        if tokens:
            ids.add(tokens[0])
    return sorted(ids)


@torch.no_grad()
def word_ranks(model, hidden: torch.Tensor, ids: Sequence[int], lens: TextLens | None = None,
               layer: int | None = None) -> torch.Tensor:
    """Rank (1 = top) of the best of `ids` among all text tokens, for each row of `hidden` ([n, d])."""
    h = hidden if lens is None else lens.transport(hidden, layer)
    norm = model.net.language_model.model.norm
    with model._autocast():
        scores = model.net.language_model.lm_head(norm(h.to(model.device, torch.bfloat16))).float()
    scores = scores[:, : model.text_vocab]  # ignore padded rows of the text head
    best = scores[:, list(ids)].max(1).values
    return (scores > best[:, None]).sum(1).cpu() + 1


def rank(m, scores: torch.Tensor, word: str) -> int:
    """Rank (1 = top) of the best first token of `word`'s spellings in one row of next-word scores."""
    scores = scores[: getattr(m, "text_vocab", scores.shape[-1])]
    best = max(float(scores[i]) for i in word_ids(m.tokenizer, word) if i < len(scores))
    return int((scores > best).sum()) + 1
