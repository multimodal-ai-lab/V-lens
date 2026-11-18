"""The V-lens transport (paper Eq. 1) and the simpler copies it was compared with: carry a hidden state h from its
extraction layer l to the insertion layer R without a fitted map.

With x = h - mu_l and N(z) = mu_R + s_R z / |z| (place a direction at the typical insertion-layer distance):
- diag:   N(D x), D_jj = sigma_R,j / sigma_l,j: each coordinate rescaled by its typical spread at the two layers
          (squared spreads floored at 1% of the layer's median coordinate variance). This is the V-lens;
- moved:  N(x), the moved copy (D = I);
- pct:    mu_R + Q_R(F_l(|x|)) x / |x|: the token's distance from its layer's average keeps its percentile ("percentile
          size" in Supp. Table S2);
- rel:    mu_R + (s_R / s_l) x: the token keeps its distance relative to the typical distance at its layer ("relative
          size").
mu_l are the layer averages over the 64 fitting passages (stored with the transport, scripts/fit_transport.py), s_R the
median distance from mu_R on 20 held-out passages (the transport's validation), sigma and the distance distributions
from the saved states of the 64 fitting and 40 further passages (scripts/collect_states.py). Nothing is fitted on
pictures.

Also here: the final normalization of each model's drawing pathway (the "final norm" variant, Supp. A.4) and the lens
ablations of Table 5 ("diag-<ablation>").
"""

from __future__ import annotations

from pathlib import Path

import torch

from vlens import paths
from vlens.paths import LENS_STORE
from vlens.models import transport_name
from vlens.transport import Transport

VARIANTS = ("moved", "diag", "pct", "rel")
FLOOR = 0.01  # of the layer's median coordinate variance

# "Final norm" (way `fnorm` of scripts/draw_cells.py): the normalization of Diffusion Lens (Toker et al., ACL 2024), where an intermediate text-encoder layer's output goes
# through the encoder's final layer norm and then to the image generator. Here: the final normalization of each model's
# image-generation pathway, with its learned gain, applied to the state h_l and inserted at R where the coordinate
# scaling's output goes. The attribute path from the model adapter (`m`) to that module:
# - bagel: the generation expert's final RMSNorm (third_party/bagel/modeling/bagel/qwen2_navit.py, Qwen2Model.norm_moe_gen;
#   in mode "gen" it normalizes the image tokens' last states before llm2vae; text tokens use .norm);
# - show_o2: the language model's final RMSNorm (third_party/show_o2/qwen2.py, Qwen2Model.norm); its output for the whole
#   sequence, text included, is what the flow head (diffusion_head_a/b) reads;
# - sensenova: the generation expert's final RMSNorm (third_party/sensenova_u1/models/neo_unify/modeling_qwen3.py,
#   Qwen3Model.norm_mot_gen; _t2i_predict_v marks every image token for it before fm_head);
# - hunyuan: no per-token final norm on the drawing path (the image tokens' last states go to final_layer, a UNetUp whose
#   first operation is a GroupNorm over the token grid); ln_f, the text head's final RMSNorm, is the closest one.
FINAL_NORM = {"bagel": "net.language_model.model.norm_moe_gen", "show_o2": "net.showo.model.norm",
              "sensenova": "net.language_model.model.norm_mot_gen", "hunyuan": "net.model.ln_f"}


def final_norm(m, model: str) -> torch.nn.Module:
    """The final normalization module of `model`'s drawing pathway, from the loaded adapter `m` (FINAL_NORM)."""
    module = m
    for attr in FINAL_NORM[model].split("."):
        module = getattr(module, attr)
    return module


def fnorm(h: torch.Tensor, norm: torch.nn.Module, spread: float | None = None,
          mu_R: torch.Tensor | None = None) -> torch.Tensor:
    """h ([d]) through `norm` as the model applies it (its own dtype, learned gain included), returned in float32 on
    the CPU. With `spread` and `mu_R`: the extra arm `fnorm:sR`, the normed state placed by the method's own N(z) =
    mu_R + s_R z / |z| (only its size and offset change)."""
    w = norm.weight
    with torch.no_grad():
        n = norm(h.detach().to(device=w.device, dtype=w.dtype)[None])[0].float().cpu()
    if spread is None:
        return n
    return mu_R.float().cpu() + n * (spread / n.norm())


def build(model: str) -> Path:
    """Compute the per-layer statistics from the saved text states (scripts/collect_states.py) and store them beside
    the transport as LENS_STORE/<model>_copy_stats/stats.pt (BAGEL, Show-o2, SenseNova-U1)."""
    t = Transport.load(transport_name(model))
    folder = paths.DATA / "states" / model
    layers = sorted(int(f.stem.split("_L")[1]) for f in folder.glob("calib_L*.pt"))
    sigma, radii = {}, {}
    for l in layers:
        h = torch.cat([torch.load(folder / f"{split}_L{l}.pt").float() for split in ("fit", "calib")])
        x = h - t.mean[l].float()
        var = x.pow(2).mean(0)
        sigma[l] = var.clamp(min=FLOOR * float(var.median())).sqrt()
        radii[l] = x.norm(dim=1).sort().values
    out = LENS_STORE / f"{model}_copy_stats"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(mean=t.mean.float(), spread=float(t.meta["spread"]), R=t.R,
                    spread_by_layer={int(k): float(v) for k, v in t.meta["spread_by_layer"].items()},
                    sigma=sigma, radii=radii, layers=layers), out / "stats.pt")
    return out


# --- Lens ablation (Table 5) ------------------------------------------------------------------------------------
# One component of the V-lens (variant "diag") changed at a time, everything else identical. Variant names
# "diag-<ablation>", ways "copy:diag-<ablation>" in scripts/draw_cells.py. With x = h - mu_l and
# D = diag(sigma_R / sigma_l), the V-lens is v = mu_R + s_R D x / |D x|.
ABLATIONS = {
    "nocenter": "x = h (the extraction-layer mean is not subtracted): v = mu_R + s_R D h / |D h|",
    "nomuR": "the insertion-layer mean is not added: v = s_R D x / |D x|",
    "nosize": "no size normalization, the rescaled deviation keeps its own size: v = mu_R + D x",
    "scalar": "one factor for every coordinate, D = (mean_j sigma_R,j / mean_j sigma_l,j) I; the factor cancels in "
              "D x / |D x|, so this is the moved copy (checked numerically, not drawn)",
    "srconly": "only the extraction-layer spreads, D = diag(1 / sigma_l): each coordinate standardized at its own layer",
    "dstonly": "only the insertion-layer spreads, D = diag(sigma_R)",
    "floor0": "no floor on the squared spreads (instead of 1% of the layer's median)",
    "floor10": "squared spreads floored at 10% of the layer's median (instead of 1%)",
    "frac8": "every statistic (mu_l, mu_R, sigma, s_R) from the first eighth of the passages' tokens",
}
# ablations that change only the statistics: built once on the CPU by ablation_stats(), stored beside the standard ones
ALT_STATS = {"floor0": dict(floor=0.0), "floor10": dict(floor=0.10), "frac8": dict(fraction=8)}


def diag_carry(h: torch.Tensor, mu_l: torch.Tensor, mu_R: torch.Tensor, sigma_l: torch.Tensor, sigma_R: torch.Tensor,
               spread: float, ablation: str = "") -> torch.Tensor:
    """The V-lens (variant "diag") for one float32 vector h ([d]) with one component ablated (ABLATIONS; "" or a
    statistics-only ablation: the V-lens itself, the same operations in the same order as CopyTransform.carry's
    "diag", so the result is bit-identical)."""
    x = h if ablation == "nocenter" else h - mu_l
    if ablation == "scalar":
        z = x * (sigma_R.mean() / sigma_l.mean())
    elif ablation == "srconly":
        z = x / sigma_l
    elif ablation == "dstonly":
        z = x * sigma_R
    else:
        z = x * (sigma_R / sigma_l)
    if ablation == "nosize":
        return mu_R + z
    if ablation == "nomuR":
        return z * (spread / z.norm())
    return mu_R + z * (spread / z.norm())


def ablation_stats(model: str, floor: float = FLOOR, fraction: int = 1) -> dict:
    """The statistics of `build()` with another variance floor and/or from the first 1/`fraction` of the tokens of every
    set of passages (in reading order: fit for mu, fit + calib for sigma, heldout for s_R, as the full statistics).
    floor=FLOOR, fraction=1 gives the standard statistics exactly (mu and s_R are then the transport's own)."""
    t = Transport.load(transport_name(model))
    folder = paths.DATA / "states" / model
    layers = sorted(int(f.stem.split("_L")[1]) for f in folder.glob("calib_L*.pt"))

    def part(split: str, l: int) -> torch.Tensor:
        h = torch.load(folder / f"{split}_L{l}.pt")
        return h[: len(h) // fraction]

    mean, spread, tokens = t.mean.float(), float(t.meta["spread"]), {}
    if fraction != 1:
        mean = mean.clone()
        for l in layers:
            mean[l] = part("fit", l).float().mean(0)
        spread = float((part("heldout", t.R).float() - mean[t.R]).norm(dim=1).median())
        tokens = {split: len(part(split, t.R)) for split in ("fit", "calib", "heldout")}
    sigma = {}
    for l in layers:
        h = torch.cat([part(split, l).float() for split in ("fit", "calib")])
        x = h - mean[l]
        var = x.pow(2).mean(0)
        sigma[l] = var.clamp(min=floor * float(var.median())).sqrt()
    return dict(mean=mean, spread=spread, R=t.R, sigma=sigma, layers=layers, floor=floor, fraction=fraction,
                tokens=tokens)


def build_ablation(model: str) -> list[Path]:
    """Store the statistics of every statistics-only ablation (ALT_STATS) as <model>_copy_stats_<ablation>/stats.pt."""
    out = []
    for name, kw in ALT_STATS.items():
        folder = LENS_STORE / f"{model}_copy_stats_{name}"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(ablation_stats(model, **kw), folder / "stats.pt")
        out.append(folder)
    return out


class CopyTransform:
    def __init__(self, model: str):
        self.s = torch.load(LENS_STORE / f"{model}_copy_stats" / "stats.pt", map_location="cpu", weights_only=False)
        self.R = self.s["R"]
        self.model, self._alt = model, {}

    def stats(self, ablation: str) -> dict:
        """The statistics an ablation uses: its own for ALT_STATS (built by build_ablation), else the standard ones."""
        if ablation not in ALT_STATS:
            return self.s
        if ablation not in self._alt:
            self._alt[ablation] = torch.load(LENS_STORE / f"{self.model}_copy_stats_{ablation}" / "stats.pt",
                                             map_location="cpu", weights_only=False)
            assert self._alt[ablation]["R"] == self.R
        return self._alt[ablation]

    def carry(self, h: torch.Tensor, layer: int, variant: str, R: int | None = None) -> torch.Tensor:
        """h ([d]) from `layer` to the insertion layer by `variant`. R: another insertion layer (the V-lens, "diag",
        only: the statistics' mean, spread per coordinate and typical distance at that layer; not used for the paper's
        numbers); default the statistics' own R."""
        if R is not None and R != self.R:
            if variant != "diag":
                raise ValueError("another insertion layer is supported for the V-lens (diag) only")
            s, h = self.s, h.float().cpu()
            z = (h - s["mean"][layer]) * (s["sigma"][R] / s["sigma"][layer])
            return s["mean"][R] + z * (float(s["spread_by_layer"][R]) / z.norm())
        s, R = self.s, self.R
        h = h.float().cpu()
        if variant.startswith("diag-"):  # the lens ablation (ABLATIONS); "diag" itself is below, unchanged
            ablation = variant.split("-", 1)[1]
            if ablation not in ABLATIONS:
                raise ValueError(f"unknown ablation {ablation!r}; known: {list(ABLATIONS)}")
            a = self.stats(ablation)
            return diag_carry(h, a["mean"][layer], a["mean"][R], a["sigma"][layer], a["sigma"][R], a["spread"], ablation)
        mu_l, mu_R = s["mean"][layer], s["mean"][R]
        x = h - mu_l
        if variant == "moved":
            return mu_R + x * (s["spread"] / x.norm())
        if variant == "diag":
            z = x * (s["sigma"][R] / s["sigma"][layer])
            return mu_R + z * (s["spread"] / z.norm())
        if variant == "pct":
            r_l, r_R = s["radii"][layer], s["radii"][R]
            p = float(torch.searchsorted(r_l, x.norm().reshape(1)).item()) / len(r_l)
            target = float(r_R[min(int(p * len(r_R)), len(r_R) - 1)])
            return mu_R + x * (target / x.norm())
        if variant == "rel":
            return mu_R + x * (s["spread"] / s["spread_by_layer"][layer])
        raise ValueError(f"unknown variant {variant!r}; known: {VARIANTS}")


def all_layer_sigma(m, s: dict, model: str) -> torch.Tensor:
    """Per-coordinate spreads at every layer ([n_layers, d]), computed exactly as build() computes them for the stored
    layers (the 64 fitting and 40 further passages, the transport's token band, the stored layer means, the 1% floor),
    from a fresh reading of the passages with the loaded model `m`; cached beside the stored statistics."""
    from vlens import prompts
    from vlens.transport import band
    import time
    cache = LENS_STORE / f"{model}_copy_stats" / "sigma_all.pt"
    if cache.exists():
        return torch.load(cache)
    pool = (prompts.PROMPTS / "text_lens_passages.txt").read_text().splitlines()
    ids = list(range(64)) + list(range(240, 280))
    n, ss = 0, None
    for pid in ids:
        r = m.read(pool[pid])
        mask = band(len(r.token_ids)).to(r.hidden[0].device)
        x = torch.stack([(h[mask].float() - s["mean"][l].to(h.device).float()) for l, h in enumerate(r.hidden)])
        ss = x.pow(2).sum(1) if ss is None else ss + x.pow(2).sum(1)
        n += int(mask.sum())
    var = ss / n
    sigma = torch.stack([v.clamp(min=FLOOR * float(v.median())).sqrt() for v in var]).cpu()
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{time.time_ns()}.tmp")  # parallel jobs: write aside, then rename at once
    torch.save(sigma, tmp)
    tmp.replace(cache)
    return sigma
