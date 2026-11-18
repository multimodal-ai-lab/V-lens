"""A miniature unified model with BAGEL's wiring, in float64, for exact CPU tests.

Text side: each layer computes keys and values from the hidden state entering it (these are cached), then
updates the hidden state with causal attention and a small MLP.
Image side: a few image tokens run through their own weights; at every layer they attend to the cached text
keys and values of that layer plus their own. A final linear map predicts the velocity, and the guess of the
finished latent is x - level * velocity, as in BAGEL.
"""

from __future__ import annotations

import math

import torch

from vlens.models.base import TextReading


def _rms(h):
    return h / h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()


class TinyUnifiedModel:
    name = "tiny"

    def __init__(self, n_layers=3, d_model=8, n_image_tokens=4, latent_dim=3, vocab=30, seed=0):
        g = torch.Generator().manual_seed(seed)

        def w(*shape):
            return torch.randn(*shape, generator=g, dtype=torch.float64) / math.sqrt(shape[0])

        self.n_layers, self.d_model, self.vocab = n_layers, d_model, vocab
        self.n_image_tokens, self.latent_dim = n_image_tokens, latent_dim
        self.device = torch.device("cpu")
        self.embed = torch.randn(vocab, d_model, generator=g, dtype=torch.float64)
        self.text = [dict(q=w(d_model, d_model), k=w(d_model, d_model), v=w(d_model, d_model),
                          o=w(d_model, d_model), up=w(d_model, 2 * d_model), down=w(2 * d_model, d_model))
                     for _ in range(n_layers)]
        self.image = [dict(q=w(d_model, d_model), k=w(d_model, d_model), v=w(d_model, d_model),
                           o=w(d_model, d_model), up=w(d_model, 2 * d_model), down=w(2 * d_model, d_model))
                      for _ in range(n_layers)]
        self.latent_in = w(latent_dim, d_model)
        self.latent_out = w(d_model, latent_dim)
        self.level_vector = torch.randn(d_model, generator=g, dtype=torch.float64)

    def _ids(self, text):
        return [1] + [2 + (ord(c) % (self.vocab - 2)) for c in text] + [0]

    def read(self, text, *, keep_grad=False, edit=None):
        ids = self._ids(text)
        h = self.embed[ids]
        if keep_grad:
            h = h.detach().requires_grad_(True)
        hidden, keys, values = [], [], []
        n = len(ids)
        causal = torch.ones(n, n, dtype=torch.bool).tril()
        with torch.set_grad_enabled(keep_grad):
            for l, p in enumerate(self.text):
                if edit is not None:
                    h = edit(l, h)
                hidden.append(h)
                x = _rms(h)
                q, k, v = x @ p["q"], x @ p["k"], x @ p["v"]
                keys.append(k)
                values.append(v)
                scores = (q @ k.T / math.sqrt(self.d_model)).masked_fill(~causal, float("-inf"))
                h = h + scores.softmax(-1) @ v @ p["o"]
                h = h + torch.tanh(_rms(h) @ p["up"]) @ p["down"]
        content = torch.ones(n, dtype=torch.bool)
        content[0] = content[-1] = False
        return TextReading(text=text, token_ids=ids, token_strs=["<s>"] + list(text) + ["</s>"],
                           hidden=hidden, content=content, cache=dict(keys=keys, values=values), last_output=h)

    def read_nothing(self):
        empty = torch.zeros(0, self.d_model, dtype=torch.float64)
        return TextReading(text="", token_ids=[], token_strs=[], hidden=[], content=torch.zeros(0, dtype=torch.bool),
                           cache=dict(keys=[empty] * self.n_layers, values=[empty] * self.n_layers))

    def noise(self, size, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(self.n_image_tokens, self.latent_dim, generator=g, dtype=torch.float64)

    def velocity(self, reading, x, level):
        z = x @ self.latent_in + level * self.level_vector
        for l, p in enumerate(self.image):
            y = _rms(z)
            q, k, v = y @ p["q"], y @ p["k"], y @ p["v"]
            keys = torch.cat([reading.cache["keys"][l], k])
            values = torch.cat([reading.cache["values"][l], v])
            z = z + (q @ keys.T / math.sqrt(self.d_model)).softmax(-1) @ values @ p["o"]
            z = z + torch.tanh(_rms(z) @ p["up"]) @ p["down"]
        return z @ self.latent_out

    def guess_finished(self, reading, x, level):
        return x - level * self.velocity(reading, x, level)

    def to_picture(self, latent):
        raise NotImplementedError
