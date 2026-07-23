"""Tests for the mask-biased attention primitive (IM-LAM Phase 1).

Run:  conda_env/bin/python tests/test_mask_biased_attention.py

The primitive adds a learnable per-head additive mask bias to attention logits,
supports separate Q/K/V sources (self- and cross-attention), upcasts the softmax
to fp32, and can optionally return attention weights. These tests pin each of
those behaviors.
"""
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ifo.common.nets.mask_biased_attention import MaskBiasedAttention


class MaskBiasedAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_output_shape_supports_cross_attention(self):
        # Q and K/V may differ in length (directed cross-attention: Nq=256, Nk=512).
        dim, heads, b, nq, nk = 32, 4, 2, 5, 7
        attn = MaskBiasedAttention(dim, heads)
        q = torch.randn(b, nq, dim)
        k = torch.randn(b, nk, dim)
        v = torch.randn(b, nk, dim)
        out = attn(q, k, v)
        self.assertEqual(out.shape, (b, nq, dim))

    def test_unbiased_matches_sdpa_reference(self):
        # With no mask bias, the module must equal standard scaled-dot-product
        # attention computed from its own Q/K/V/out projections.
        dim, heads, b, n = 32, 4, 2, 6
        attn = MaskBiasedAttention(dim, heads)
        q_in = torch.randn(b, n, dim)
        k_in = torch.randn(b, n, dim)
        v_in = torch.randn(b, n, dim)
        out = attn(q_in, k_in, v_in, mask_bias=None)

        hd = dim // heads

        def split(t):
            return t.view(b, -1, heads, hd).transpose(1, 2)

        q = split(attn.q_proj(q_in))
        k = split(attn.k_proj(k_in))
        v = split(attn.v_proj(v_in))
        ref_ctx = F.scaled_dot_product_attention(q, k, v)
        ref = attn.out_proj(ref_ctx.transpose(1, 2).reshape(b, n, dim))
        self.assertTrue(torch.allclose(out, ref, atol=1e-5, rtol=1e-4))

    def test_none_bias_equals_zero_bias(self):
        # A zeros mask bias must be a no-op, identical to passing None.
        dim, heads, b, n = 32, 4, 2, 6
        attn = MaskBiasedAttention(dim, heads)
        q = torch.randn(b, n, dim)
        k = torch.randn(b, n, dim)
        v = torch.randn(b, n, dim)
        out_none = attn(q, k, v, mask_bias=None)
        out_zero = attn(q, k, v, mask_bias=torch.zeros(b, n))
        self.assertTrue(torch.allclose(out_none, out_zero, atol=1e-6))

    def test_large_beta_collapses_onto_masked_key(self):
        # beta -> large with a one-hot occupancy recovers hard masked attention:
        # all attention mass moves onto the masked key.
        dim, heads, b, n = 32, 4, 2, 8
        attn = MaskBiasedAttention(dim, heads)
        with torch.no_grad():
            attn.beta.fill_(50.0)
        q = torch.randn(b, n, dim)
        k = torch.randn(b, n, dim)
        v = torch.randn(b, n, dim)
        j0 = 3
        mask_bias = torch.zeros(b, n)
        mask_bias[:, j0] = 1.0
        _, aw = attn(q, k, v, mask_bias=mask_bias, return_attn=True)
        self.assertTrue(torch.allclose(aw[..., j0], torch.ones(b, heads, n), atol=1e-3))

    def test_return_attn_contract(self):
        # Default call returns a bare tensor; return_attn=True adds normalized weights.
        dim, heads, b, nq, nk = 32, 4, 2, 5, 7
        attn = MaskBiasedAttention(dim, heads)
        q = torch.randn(b, nq, dim)
        k = torch.randn(b, nk, dim)
        v = torch.randn(b, nk, dim)
        out_only = attn(q, k, v)
        self.assertIsInstance(out_only, torch.Tensor)
        _, aw = attn(q, k, v, return_attn=True)
        self.assertEqual(aw.shape, (b, heads, nq, nk))
        self.assertTrue(torch.allclose(aw.sum(-1), torch.ones(b, heads, nq), atol=1e-5))

    def test_fp32_softmax_under_bf16_autocast(self):
        # The softmax fp32 upcast must run cleanly under bf16 autocast (training regime).
        dim, heads, b, n = 32, 4, 2, 6
        attn = MaskBiasedAttention(dim, heads)
        q = torch.randn(b, n, dim)
        k = torch.randn(b, n, dim)
        v = torch.randn(b, n, dim)
        mask_bias = torch.rand(b, n)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = attn(q, k, v, mask_bias=mask_bias)
        self.assertTrue(torch.isfinite(out).all())

    def test_beta_receives_gradient(self):
        # The learnable bias must be in the graph so it can be trained/logged.
        dim, heads, b, n = 32, 4, 2, 6
        attn = MaskBiasedAttention(dim, heads)
        q = torch.randn(b, n, dim)
        k = torch.randn(b, n, dim)
        v = torch.randn(b, n, dim)
        mask_bias = torch.rand(b, n)
        out = attn(q, k, v, mask_bias=mask_bias)
        out.sum().backward()
        self.assertIsNotNone(attn.beta.grad)
        self.assertGreater(attn.beta.grad.abs().sum().item(), 0.0)

    def test_beta_initialized_to_two(self):
        # Proposal: per-head beta initialized to 2.0.
        attn = MaskBiasedAttention(32, 4)
        self.assertEqual(tuple(attn.beta.shape), (4,))
        self.assertTrue(torch.allclose(attn.beta.detach(), torch.full((4,), 2.0)))


if __name__ == "__main__":
    unittest.main()
