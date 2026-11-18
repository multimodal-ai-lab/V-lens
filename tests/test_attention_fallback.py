"""The plain attention used for full-precision checks must match flash-attn's conventions:
grouped key/value heads, scale 1/sqrt(head_dim), causal mask aligned to the bottom right, packed sequences."""

import math

import pytest
import torch

# needs BAGEL's patched code (third_party/setup.sh, third_party on PYTHONPATH) and flash-attn installed
qwen2_navit = pytest.importorskip("bagel.modeling.bagel.qwen2_navit")


def naive(q, k, v, cu_q, cu_k, causal):
    heads, kv_heads, dim = q.shape[1], k.shape[1], q.shape[2]
    out = torch.empty_like(q)
    for i in range(len(cu_q) - 1):
        qs, ks, vs = q[cu_q[i]:cu_q[i + 1]], k[cu_k[i]:cu_k[i + 1]], v[cu_k[i]:cu_k[i + 1]]
        n_q, n_k = qs.shape[0], ks.shape[0]
        for h in range(heads):
            g = h // (heads // kv_heads)
            scores = qs[:, h] @ ks[:, g].T / math.sqrt(dim)
            if causal:
                for a in range(n_q):
                    for b in range(n_k):
                        if b > a + (n_k - n_q):
                            scores[a, b] = float("-inf")
            out[cu_q[i]:cu_q[i + 1], h] = scores.softmax(-1) @ vs[:, g]
    return out


def test_fallback_matches_naive_attention():
    qwen2_navit.COMPUTE_DTYPE = torch.float32
    try:
        g = torch.Generator().manual_seed(0)
        lens_q, lens_k = [5, 1, 4], [5, 7, 9]  # equal lengths, one query on a long cache, and more keys than queries
        cu_q = torch.tensor([0] + list(torch.tensor(lens_q).cumsum(0)))
        cu_k = torch.tensor([0] + list(torch.tensor(lens_k).cumsum(0)))
        q = torch.randn(sum(lens_q), 4, 8, generator=g)
        k = torch.randn(sum(lens_k), 2, 8, generator=g)
        v = torch.randn(sum(lens_k), 2, 8, generator=g)
        for causal in (True, False):
            got = qwen2_navit.varlen_attention(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k), causal)
            assert torch.allclose(got, naive(q, k, v, cu_q, cu_k, causal), atol=1e-5), causal
    finally:
        qwen2_navit.COMPUTE_DTYPE = torch.bfloat16
