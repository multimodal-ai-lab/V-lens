"""Small LoRA adapters on the text (understanding) weights, for lens tuning (scripts/lens_tuning.py,
scripts/state_tuning.py).

The adapters sit on the attention and MLP projections of the understanding expert in the first layers only, so they can
change the hidden state of a text token before it reaches the V-lens insertion layer while the generation weights stay
untouched. They have one global switch: lens tuning turns them on only while the source text is read, and off while the
drawing request "a photo of a thing" is read and while the picture is drawn (the understanding weights also process the
request and the image markers, so an adapter left on could improve the request instead of the source word).

    adapters = add_lora(model, layers=range(8), rank=8, alpha=16)
    with adapters.on():
        src = model.read(text, keep_grad=True)
"""

from __future__ import annotations

import contextlib
import math

import torch
from torch import nn

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, switch: list):
        super().__init__()
        self.base = base
        self.switch = switch
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        if not self.switch[0]:
            return y
        return y + (x.to(self.A.dtype) @ self.A.T @ self.B.T * self.scale).to(y.dtype)


class Adapters:
    def __init__(self, modules: dict, switch: list):
        self.modules = modules
        self.switch = switch

    def parameters(self):
        return [p for m in self.modules.values() for p in (m.A, m.B)]

    @contextlib.contextmanager
    def on(self):
        old = self.switch[0]
        self.switch[0] = True
        try:
            yield
        finally:
            self.switch[0] = old

    def state_dict(self) -> dict:
        return {f"{name}.{k}": getattr(m, k).detach().cpu() for name, m in self.modules.items() for k in ("A", "B")}

    def load_state_dict(self, state: dict):
        for name, m in self.modules.items():
            for k in ("A", "B"):
                getattr(m, k).data.copy_(state[f"{name}.{k}"].to(getattr(m, k).device))


# HunyuanImage-3.0: fused q/k/v, and a mixture of experts whose always-on shared MLP takes the adapters
HUNYUAN_ATTENTION = ("qkv_proj", "o_proj")
HUNYUAN_MLP = ("gate_and_up_proj", "down_proj")


def add_lora(model, layers, rank: int = 8, alpha: float = 16.0, seed: int = 0) -> Adapters:
    """Wrap the understanding projections of `layers` (attention q/k/v/o and MLP gate/up/down) in switchable adapters,
    off by default. B starts at zero, so the model is unchanged until training moves it."""
    torch.manual_seed(seed)
    switch = [False]
    modules = {}
    for i in layers:
        layer = model.layers[i]
        if hasattr(layer.self_attn, "qkv_proj"):
            groups = ((layer.self_attn, "self_attn", HUNYUAN_ATTENTION), (layer.mlp.shared_mlp, "mlp.shared_mlp", HUNYUAN_MLP))
        else:
            groups = ((layer.self_attn, "self_attn", ATTENTION), (layer.mlp, "mlp", MLP))
        for parent, where, names in groups:
            for n in names:
                wrapped = LoRALinear(getattr(parent, n), rank, alpha, switch)
                setattr(parent, n, wrapped)
                modules[f"layers.{i}.{where}.{n}"] = wrapped
    return Adapters(modules, switch)
