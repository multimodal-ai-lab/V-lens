"""The final-norm variant (vlens/carry.py: fnorm, final_norm; Supp. A.4), on the CPU."""

import types

import torch

from vlens.carry import FINAL_NORM, final_norm, fnorm


class RMSNorm(torch.nn.Module):
    """The models' final RMSNorm (Qwen2RMSNorm / Qwen3RMSNorm / HunyuanRMSNorm): float32 inside, gain in the model dtype."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(d) * 3)
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def test_fnorm_is_the_module_with_its_gain():
    torch.manual_seed(0)
    norm = RMSNorm(64).to(torch.bfloat16)
    h = torch.randn(64) * 40 + 3
    out = fnorm(h, norm)
    assert out.dtype == torch.float32 and out.shape == (64,)
    assert torch.equal(out, norm(h.to(torch.bfloat16)[None])[0].float())
    # without the gain the result would have RMS 1; with it, coordinate j is scaled by weight_j
    plain = h / h.pow(2).mean().sqrt()
    assert torch.allclose(out, norm.weight.float() * plain, rtol=2e-2, atol=2e-2)
    # the normalization removes the state's size: a state ten times larger gives the same output
    assert torch.allclose(fnorm(h * 10, norm), out, rtol=1e-2, atol=1e-2)


def test_fnorm_at_s_R_places_the_normed_direction():
    torch.manual_seed(1)
    norm = RMSNorm(32).to(torch.bfloat16)
    h, mu_R = torch.randn(32) * 5, torch.randn(32) * 7
    n = fnorm(h, norm)
    v = fnorm(h, norm, 123.0, mu_R)
    assert torch.allclose((v - mu_R).norm(), torch.tensor(123.0), rtol=1e-5)
    assert torch.nn.functional.cosine_similarity(v - mu_R, n, dim=0) > 0.9999


def test_final_norm_paths():
    """Every path of FINAL_NORM resolves on an object with the adapters' attribute layout."""
    norm = RMSNorm(8)
    ns = types.SimpleNamespace
    fake = {"bagel": ns(net=ns(language_model=ns(model=ns(norm_moe_gen=norm)))),
            "show_o2": ns(net=ns(showo=ns(model=ns(norm=norm)))),
            "sensenova": ns(net=ns(language_model=ns(model=ns(norm_mot_gen=norm)))),
            "hunyuan": ns(net=ns(model=ns(ln_f=norm)))}
    assert set(fake) == set(FINAL_NORM)
    for model, m in fake.items():
        assert final_norm(m, model) is norm
