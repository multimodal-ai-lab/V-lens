"""The small interface every unified model implements.

A "unified model" here reads text with a transformer and draws an image by denoising, and its image side
reads the hidden states of the text. the lens needs only a few things from such a model:

- read a text and hand back the hidden state entering every layer, for every token
  (optionally keeping the computation graph, so gradients can flow back into those hidden states);
- optionally change those hidden states while the text is being read (an "edit");
- predict, for an image at some noise level, the model's guess of the finished image;
- turn an image latent into a picture.

Noise level convention used everywhere in this code: 1.0 means pure noise, 0.0 means a finished image.
Each adapter converts to its model's own convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import torch
from PIL import Image

# An edit changes the hidden state entering a layer while the text is read:
#   edit(layer_index, hidden) -> new_hidden, where hidden has shape [n_tokens, d_model].
Edit = Callable[[int, torch.Tensor], torch.Tensor]


@dataclass
class TextReading:
    """Everything the model computed while reading one piece of text."""

    text: str
    token_ids: list[int]
    token_strs: list[str]
    # hidden[l] is the hidden state entering layer l, shape [n_tokens, d_model].
    hidden: list[torch.Tensor]
    # True for tokens that carry the text itself; False for start/end markers.
    content: torch.Tensor
    # Whatever the image side reads (for BAGEL: the key/value cache plus position bookkeeping).
    cache: Any
    # The final hidden state after the last layer and final norm, if the model has a text head.
    final: Optional[torch.Tensor] = None
    # The output of the last layer before the final norm (what a text J-lens transports to).
    last_output: Optional[torch.Tensor] = None

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


class UnifiedModel(Protocol):
    name: str
    n_layers: int
    d_model: int
    device: torch.device

    def read(self, text: str, *, keep_grad: bool = False, edit: Optional[Edit] = None) -> TextReading:
        """Read `text`, recording the hidden state entering every layer."""

    def read_prompt(self, text: str, *, keep_grad: bool = False, edit: Optional[Edit] = None) -> TextReading:
        """Read a drawing prompt in the form draw() needs (for most models the same as read())."""

    def read_nothing(self) -> TextReading:
        """The empty prompt, used for guidance and as a baseline."""

    def noise(self, size: int, seed: int) -> torch.Tensor:
        """Pure-noise starting latent for a size x size picture."""

    def velocity(self, reading: TextReading, x: torch.Tensor, level: float) -> torch.Tensor:
        """One pass of the image side (no guidance), in the model's own output convention."""

    def guess_finished(self, reading: TextReading, x: torch.Tensor, level: float) -> torch.Tensor:
        """The model's guess of the finished image latent, given latent `x` at noise `level`."""

    def to_picture(self, latent: torch.Tensor) -> Image.Image:
        """Decode an image latent into a picture."""
