"""Transport maps between layers at the same token (vlens/transport.py), on the tiny model in float64."""

import torch

from tiny_model import TinyUnifiedModel
from vlens.hidden import add_at
from vlens.transport import Transport, passage_rows


def brute_force(model, text, source_layer, target_layer, p):
    """d h_target[p] / d h_source[p] by differentiating through an edit at one token."""
    def f(delta):
        return model.read(text, keep_grad=True, edit=add_at(source_layer, [p], delta)).hidden[target_layer][p]
    return torch.autograd.functional.jacobian(f, torch.zeros(model.d_model, dtype=torch.float64))


def test_single_token_rows_are_exact():
    """With one token in the band there are no cross-token terms, so the signed estimate is exact."""
    model = TinyUnifiedModel()
    text = "the spider web"
    reading = model.read(text)
    mask = torch.zeros(reading.n_tokens, dtype=torch.bool)
    mask[4] = True
    r = passage_rows(model, text, reading_layer=2, backward_targets=[], seed=0, mask=mask)
    for l in (0, 1):
        assert torch.allclose(r["forward"][l].double(), brute_force(model, text, l, 2, 4), atol=1e-6)
    r = passage_rows(model, text, reading_layer=1, backward_targets=[2], seed=0, mask=mask)
    assert torch.allclose(r["backward"][2].double(), brute_force(model, text, 1, 2, 4), atol=1e-6)


def test_backward_map_inverts():
    """The backward transport solves B v = h - mean, so applying B afterwards gives back the input."""
    g = torch.Generator().manual_seed(0)
    d = 6
    B = torch.eye(d) + 0.3 * torch.randn(d, d, generator=g)
    mean = torch.randn(3, d, generator=g)
    t = Transport(reading_layer=1, forward=torch.eye(d)[None], backward={2: B}, mean=mean)
    h = torch.randn(d, generator=g)
    v = t.to_reading_layer(h, 2, ridge_fraction=1e-9)
    assert torch.allclose((v - mean[1]) @ B.T, h - mean[2], atol=1e-4)
    assert torch.allclose(t.to_reading_layer(mean[0], 0), mean[1])


def test_size_modes():
    torch.manual_seed(0)
    d, R = 6, 2
    t = Transport(R, torch.randn(R, d, d, dtype=torch.float32), {}, torch.randn(4, d),
                  meta=dict(spread=3.0, spread_by_layer={0: 2.0, 1: 5.0, 2: 3.0}))
    h = t.mean[0] + torch.randn(5, d)
    raw = t.to_reading_layer(h, 0)
    assert torch.allclose(t.to_reading_layer(h, 0, match_size=True, size="none"), raw)
    fixed = t.to_reading_layer(h, 0, match_size=True) - t.mean[R]
    assert torch.allclose(fixed.norm(dim=-1), torch.full((5,), 3.0), atol=1e-5)
    rel = t.to_reading_layer(h, 0, match_size=True, size="relative") - t.mean[R]
    assert torch.allclose(rel.norm(dim=-1), 3.0 * (h - t.mean[0]).norm(dim=-1) / 2.0, atol=1e-4)
    assert torch.allclose(torch.nn.functional.cosine_similarity(rel, fixed, dim=-1), torch.ones(5), atol=1e-5)
