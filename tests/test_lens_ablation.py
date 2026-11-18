"""The lens ablation of Table 5 (vlens/carry.py: ABLATIONS, diag_carry, ablation_stats), on the CPU with made-up
statistics: every ablation changes exactly its one component, and with nothing ablated the V-lens ("diag") comes out
bit for bit."""

import types

import pytest
import torch

from vlens import carry as ct

D, N_LAYERS, R = 48, 6, 4


def fake_stats(seed=0):
    g = torch.Generator().manual_seed(seed)
    mean = torch.randn(N_LAYERS, D, generator=g) * 20
    sigma = {l: torch.rand(D, generator=g) * 3 + 0.2 for l in range(N_LAYERS)}
    return dict(mean=mean, spread=37.5, R=R, sigma=sigma, spread_by_layer={l: 10.0 + l for l in range(N_LAYERS)},
                layers=list(range(N_LAYERS)))


def transform(stats, alt=None):
    c = ct.CopyTransform.__new__(ct.CopyTransform)
    c.s, c.R, c.model, c._alt = stats, stats["R"], "fake", dict(alt or {})
    return c


@pytest.fixture
def setup():
    s = fake_stats()
    h = torch.randn(D, generator=torch.Generator().manual_seed(1)) * 15 + 4
    return s, h, 2


def test_no_ablation_is_the_v_lens_bit_for_bit(setup):
    s, h, l = setup
    c = transform(s, alt={a: s for a in ct.ALT_STATS})  # statistics-only ablations with the standard statistics
    v = c.carry(h, l, "diag")
    assert torch.equal(ct.diag_carry(h, s["mean"][l], s["mean"][R], s["sigma"][l], s["sigma"][R], s["spread"]), v)
    for a in ct.ALT_STATS:
        assert torch.equal(c.carry(h, l, f"diag-{a}"), v)


def test_each_ablation_changes_its_component(setup):
    s, h, l = setup
    c = transform(s)
    mu_l, mu_R, sl, sR, sp = s["mean"][l], s["mean"][R], s["sigma"][l], s["sigma"][R], s["spread"]
    x = h - mu_l
    Dx, Dh = x * sR / sl, h * sR / sl
    close = lambda a, b: torch.allclose(a, b, rtol=1e-5, atol=1e-4)  # noqa: E731
    v = c.carry(h, l, "diag")
    assert close(v, mu_R + sp * Dx / Dx.norm())
    assert close(c.carry(h, l, "diag-nocenter"), mu_R + sp * Dh / Dh.norm())
    assert close(c.carry(h, l, "diag-nomuR"), sp * Dx / Dx.norm())
    assert close(c.carry(h, l, "diag-nomuR"), v - mu_R)
    assert close(c.carry(h, l, "diag-nosize"), mu_R + Dx)
    assert close(c.carry(h, l, "diag-srconly"), mu_R + sp * (x / sl) / (x / sl).norm())
    assert close(c.carry(h, l, "diag-dstonly"), mu_R + sp * (x * sR) / (x * sR).norm())
    # one factor for all coordinates cancels in D x / |D x|: the moved copy
    assert close(c.carry(h, l, "diag-scalar"), c.carry(h, l, "moved"))
    for a in ("nocenter", "nomuR", "nosize", "srconly", "dstonly"):
        assert not close(c.carry(h, l, f"diag-{a}"), v), a
    # what each keeps: the size s_R around mu_R (all but nomuR and nosize), the content direction (nomuR)
    for a in ("nocenter", "srconly", "dstonly", "scalar"):
        assert abs(float((c.carry(h, l, f"diag-{a}") - mu_R).norm()) - sp) < 1e-3
    with pytest.raises(ValueError):
        c.carry(h, l, "diag-unknown")


def test_statistics_only_ablations_use_their_own_statistics(setup):
    s, h, l = setup
    other = fake_stats(seed=5)
    c = transform(s, alt={"frac8": other})
    v = c.carry(h, l, "diag-frac8")
    assert torch.equal(v, ct.diag_carry(h, other["mean"][l], other["mean"][R], other["sigma"][l], other["sigma"][R],
                                        other["spread"]))
    assert not torch.allclose(v, c.carry(h, l, "diag"))


def test_ablation_stats_reproduce_build_and_change_only_floor_or_fraction(tmp_path, monkeypatch):
    """ablation_stats(FLOOR, 1) gives build()'s statistics exactly; floor and fraction change only what they should."""
    g = torch.Generator().manual_seed(3)
    layers = [0, 2, R]
    states = tmp_path / "states" / "fake"
    states.mkdir(parents=True)
    scale = torch.ones(D)
    scale[:3] = 0.05  # three coordinates with a tiny spread, so the floors bind
    for l in layers:
        for split, n in (("fit", 160), ("calib", 80), ("heldout", 64)):
            torch.save((torch.randn(n, D, generator=g) * 5 * scale + l).to(torch.float16), states / f"{split}_L{l}.pt")
    mean = torch.zeros(N_LAYERS, D)
    for l in layers:
        mean[l] = torch.load(states / f"fit_L{l}.pt").float().mean(0) + 0.01  # the transport's own (float32) mean
    t = types.SimpleNamespace(mean=mean, R=R, meta=dict(spread=12.0, spread_by_layer={l: 1.0 for l in layers}))
    monkeypatch.setattr(ct, "Transport", types.SimpleNamespace(load=lambda name: t))
    monkeypatch.setattr(ct, "transport_name", lambda m: m)
    monkeypatch.setattr(ct, "paths", types.SimpleNamespace(DATA=tmp_path))
    monkeypatch.setattr(ct, "LENS_STORE", tmp_path / "lenses")
    std = torch.load(ct.build("fake") / "stats.pt", weights_only=False)
    same = ct.ablation_stats("fake")
    assert torch.equal(same["mean"], std["mean"]) and same["spread"] == std["spread"]
    assert all(torch.equal(same["sigma"][l], std["sigma"][l]) for l in layers)
    # floors: the tiny coordinates are lifted by 1% (standard) and 10%, untouched without a floor; the rest never moves
    f0, f10 = ct.ablation_stats("fake", floor=0.0), ct.ablation_stats("fake", floor=0.10)
    for l in layers:
        var = (torch.cat([torch.load(states / f"{s}_L{l}.pt").float() for s in ("fit", "calib")]) - mean[l]).pow(2).mean(0)
        assert torch.equal(f0["sigma"][l], var.sqrt())
        assert torch.allclose(f10["sigma"][l][:3], (0.10 * var.median()).sqrt().expand(3))
        assert torch.equal(f10["sigma"][l][3:], std["sigma"][l][3:]) and torch.equal(f0["sigma"][l][3:], std["sigma"][l][3:])
        assert torch.equal(f0["mean"], std["mean"]) and f0["spread"] == std["spread"]
    # an eighth: the first 20 fit, 10 calib and 8 held-out tokens, every statistic recomputed from them
    e = ct.ablation_stats("fake", fraction=8)
    assert e["tokens"] == dict(fit=20, calib=10, heldout=8)
    for l in layers:
        fit = torch.load(states / f"fit_L{l}.pt")[:20].float()
        assert torch.allclose(e["mean"][l], fit.mean(0))
        h = torch.cat([fit, torch.load(states / f"calib_L{l}.pt")[:10].float()])
        var = (h - e["mean"][l]).pow(2).mean(0)
        assert torch.allclose(e["sigma"][l], var.clamp(min=ct.FLOOR * float(var.median())).sqrt())
    held = torch.load(states / f"heldout_L{R}.pt")[:8].float()
    assert e["spread"] == pytest.approx(float((held - e["mean"][R]).norm(dim=1).median()))
    # build_ablation stores one folder per statistics-only ablation, read back by CopyTransform.stats
    ct.build_ablation("fake")
    c = ct.CopyTransform("fake")
    for a in ct.ALT_STATS:
        assert c.stats(a)["floor"] == ct.ALT_STATS[a].get("floor", ct.FLOOR)
    assert c.stats("nocenter") is c.s
