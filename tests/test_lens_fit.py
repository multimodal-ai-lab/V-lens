"""Exact CPU tests of the average-Jacobian fitter (vlens/lens_fit.py) on the tiny model (float64).

The brute-force reference differentiates a target with respect to a perturbation added at one layer and one token,
separately for every (layer, token) pair, through the model's own edit mechanism. The fitter must reproduce the
average over tokens of those Jacobians, for every layer, from one backward pass per output direction. The generic
tests use the tiny model's guess of a finished latent as target; the last test uses the text J-lens target.
"""

import torch

from tiny_model import TinyUnifiedModel
from vlens import lens_fit
from vlens.hidden import add_at, find_word

PROMPTS = ["a cat", "the red fox", "spider web", "blue sky at noon", "an old boat", "two dogs"]


def setup():
    model = TinyUnifiedModel()
    start = model.noise(0, seed=7)
    def target(m, reading, prompt_index):
        return m.guess_finished(reading, start, 0.8).flatten()

    return model, start, target


def brute_force_rows(model, text, target):
    """Average over content tokens of d target / d h_l[p], one (layer, token) pair at a time."""
    reading = model.read(text)
    rows = []
    for l in range(model.n_layers):
        per_token = []
        for p in torch.nonzero(reading.content).flatten().tolist():
            def f(delta, l=l, p=p):
                return target(model, model.read(text, keep_grad=True, edit=add_at(l, [p], delta)), 0)
            per_token.append(torch.autograd.functional.jacobian(f, torch.zeros(model.d_model, dtype=torch.float64)))
        rows.append(torch.stack(per_token).mean(0))
    return torch.stack(rows)


def test_rows_match_brute_force():
    model, _, target = setup()
    for text in ["a cat", "spider web"]:
        fitted = lens_fit.prompt_rows(model, text, target)["rows"].double()
        reference = brute_force_rows(model, text, target)
        assert torch.allclose(fitted, reference, atol=1e-6, rtol=1e-6), (fitted - reference).abs().max()


def test_every_layer_reaches_the_picture():
    model, _, target = setup()
    rows = lens_fit.prompt_rows(model, "a cat", target)["rows"]
    assert all(rows[l].abs().max() > 0 for l in range(model.n_layers))


def test_basis_rows_are_projected_full_rows():
    model, _, target = setup()
    full = lens_fit.prompt_rows(model, "the red fox", target)["rows"].double()
    q, _ = torch.linalg.qr(torch.randn(full.shape[1], 5, dtype=torch.float64))
    basis = q.T  # 5 orthonormal directions in latent space
    projected = lens_fit.prompt_rows(model, "the red fox", target, basis)["rows"].double()
    for l in range(model.n_layers):
        assert torch.allclose(projected[l], basis @ full[l], atol=1e-6)


def test_selected_layers_match_all_layers():
    model, _, target = setup()
    all_rows = lens_fit.prompt_rows(model, "an old boat", target)["rows"]
    one = lens_fit.prompt_rows(model, "an old boat", target, layers=[1])["rows"]
    assert torch.allclose(one[0], all_rows[1], atol=1e-12)


def test_shard_merge_equals_single_fit():
    model, _, target = setup()
    log = lambda s: None
    whole = lens_fit.finish(lens_fit.fit(model, PROMPTS, target, log=log))
    a = lens_fit.fit(model, PROMPTS[:2], target, prompt_ids=[0, 1], log=log)
    b = lens_fit.fit(model, PROMPTS[2:], target, prompt_ids=[2, 3, 4, 5], log=log)
    merged = lens_fit.finish(lens_fit.merge([b, a]))
    for key in ("rows", "mean_hidden", "mean_guess"):
        assert torch.allclose(merged[key], whole[key], atol=1e-6), key
    assert merged["n_prompts"] == len(PROMPTS)


def test_merge_refuses_overlap():
    model, _, target = setup()
    a = lens_fit.fit(model, PROMPTS[:2], target, prompt_ids=[0, 1], log=lambda s: None)
    try:
        lens_fit.merge([a, a])
    except ValueError:
        return
    raise AssertionError("overlapping shards were merged")


def test_resume_from_checkpoint(tmp_path):
    model, _, target = setup()
    log = lambda s: None
    ckpt = tmp_path / "fit.pt"
    lens_fit.fit(model, PROMPTS[:3], target, checkpoint=ckpt, log=log)  # a job that stopped early
    resumed = lens_fit.finish(lens_fit.fit(model, PROMPTS, target, checkpoint=ckpt, log=log))
    whole = lens_fit.finish(lens_fit.fit(model, PROMPTS, target, log=log))
    assert torch.allclose(resumed["rows"], whole["rows"], atol=1e-6)
    assert resumed["n_prompts"] == len(PROMPTS)


def test_find_word():
    model = TinyUnifiedModel()
    reading = model.read("the spider spins")
    positions = find_word(reading, "spider")
    assert "".join(reading.token_strs[p] for p in positions) == "spider"


def test_text_lens_rows_match_brute_force():
    """The text J-lens estimator: target = last-layer output summed over a band of tokens, sources = that band.
    Causal attention makes each source token's gradient the sum over itself and later tokens."""
    model = TinyUnifiedModel()
    text = "the spider web"

    def band(reading):
        mask = torch.zeros(reading.n_tokens, dtype=torch.bool)
        mask[2:reading.n_tokens - 1] = True
        return mask

    def target(m, reading, i):
        return reading.last_output[band(reading)].sum(0)

    fitted = lens_fit.prompt_rows(model, text, target, sources=band)["rows"].double()
    reading = model.read(text)
    positions = torch.nonzero(band(reading)).flatten().tolist()
    for l in range(model.n_layers):
        per_token = []
        for p in positions:
            f = lambda delta, l=l, p=p: target(model, model.read(text, keep_grad=True, edit=add_at(l, [p], delta)), 0)
            per_token.append(torch.autograd.functional.jacobian(f, torch.zeros(model.d_model, dtype=torch.float64)))
        assert torch.allclose(fitted[l], torch.stack(per_token).mean(0), atol=1e-6), l
