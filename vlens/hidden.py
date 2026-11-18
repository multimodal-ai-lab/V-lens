"""Reading and changing hidden states.

An edit is a function edit(layer, hidden) -> hidden that a model applies to the hidden state entering a
layer while it reads a text (see models/base.py). The helpers here build the edits used in the
experiments: adding a vector at some tokens ("push") and replacing the hidden state at some tokens
("activation patching": copying a hidden state from one text into another).
"""

from __future__ import annotations

from typing import Sequence

import torch

from vlens.models.base import Edit, TextReading


def add_at(layer: int, positions: Sequence[int], vector: torch.Tensor) -> Edit:
    """Add `vector` ([d_model]) to the hidden state entering `layer` at the given token positions."""
    positions = list(positions)

    def edit(layer_index: int, hidden: torch.Tensor) -> torch.Tensor:
        if layer_index != layer:
            return hidden
        hidden = hidden.clone()
        hidden[positions] = hidden[positions] + vector.to(hidden.dtype).to(hidden.device)
        return hidden

    return edit


def replace_at(layer: int, positions: Sequence[int], vectors: torch.Tensor) -> Edit:
    """Replace the hidden state entering `layer` at `positions` with `vectors` ([len(positions), d_model])."""
    positions = list(positions)

    def edit(layer_index: int, hidden: torch.Tensor) -> torch.Tensor:
        if layer_index != layer:
            return hidden
        hidden = hidden.clone()
        hidden[positions] = vectors.to(hidden.dtype).to(hidden.device)
        return hidden

    return edit


def combine(*edits: Edit) -> Edit:
    def edit(layer_index: int, hidden: torch.Tensor) -> torch.Tensor:
        for e in edits:
            hidden = e(layer_index, hidden)
        return hidden

    return edit


def find_word(reading: TextReading, word: str, occurrence: int = -1) -> list[int]:
    """Token positions covering `word` in the reading's text (default: its last occurrence).

    Works on the decoded token strings, so it also finds a word that was split into several tokens.
    """
    starts, text = [], ""
    for s in reading.token_strs:
        starts.append(len(text))
        text += s
    hits, i = [], text.find(word)
    while i != -1:
        hits.append(i)
        i = text.find(word, i + 1)
    if not hits:
        raise ValueError(f"{word!r} not found in {text!r}")
    begin = hits[occurrence]
    end = begin + len(word)
    return [p for p, s in enumerate(starts)
            if s < end and s + len(reading.token_strs[p]) > begin]


def content_mean(hidden: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
    """Average hidden state over the content tokens of one text, in float32."""
    return hidden[content.to(hidden.device)].float().mean(0)
