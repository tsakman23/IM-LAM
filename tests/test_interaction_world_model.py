"""Tests for the IM-LAM FDM (InteractionWorldModel, Phase 3).

Run:  conda_env/bin/python tests/test_interaction_world_model.py

The FDM is an IMPALA encoder -> interaction bottleneck -> IMPALA decoder. Unlike
MaskLAM's ImpalaWorldModel, z_t is not concatenated at the bottleneck (it enters via
F_A), so the decoder is built from the un-doubled bottleneck width, and at init the
zero-init write-back makes the whole FDM action-blind (predicts o_{t+1} from o_t alone).
Tests use a small 32x32 input (3 encoder blocks -> 4x4 bottleneck) for speed.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ifo.common.nets.interaction_world_model import InteractionWorldModel


def _model(**kw):
    # Small config: input 32x32, encoder_channels (16,32,32) at channel_multiplier 1 => 4x4 x 32 bottleneck.
    defaults = dict(
        shape=(15, 32, 32),  # frame_stack*(c+2) = 3*5 input channels
        output_channels=3,
        condition_dim=8,
        channel_multiplier=1,
        encoder_channels=(16, 32, 32),
        encoder_num_res_blocks=2,
        num_heads=4,
    )
    defaults.update(kw)
    shape = defaults.pop("shape")
    return InteractionWorldModel(shape, **defaults)


class InteractionWorldModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.b = 2
        self.model = _model()
        self.x = torch.randn(self.b, 15, 32, 32)
        self.z = torch.randn(self.b, 8)
        self.m_a = (torch.rand(self.b, 1, 32, 32) > 0.5).float()
        self.m_o = (torch.rand(self.b, 1, 32, 32) > 0.5).float()

    def test_final_encoder_shape(self):
        # 3 encoder blocks halve 32 -> 16 -> 8 -> 4; width 32 * channel_multiplier(1) = 32.
        self.assertEqual(self.model.final_encoder_shape, (32, 4, 4))

    def test_forward_shape(self):
        y = self.model(self.x, self.z, self.m_a, self.m_o)
        self.assertEqual(tuple(y.shape), (self.b, 3, 32, 32))

    def test_output_in_tanh_half_range(self):
        y = self.model(self.x, self.z, self.m_a, self.m_o)
        self.assertTrue(torch.all(y >= -0.5) and torch.all(y <= 0.5))

    def test_decoder_built_from_undoubled_width(self):
        # z_t is NOT concatenated at the bottleneck, so the decoder's first block takes d, not 2d.
        self.assertEqual(self.model.decoder[0].in_channels, self.model.final_encoder_shape[0])  # 32, not 64

    def test_world_model_has_no_act_proj(self):
        # z_t enters through F_A's own act_proj, not a channel-wise concat here.
        self.assertFalse(hasattr(self.model, "act_proj"))

    def test_num_res_blocks_threaded_into_f_a(self):
        # F_A's residual depth must equal encoder_num_res_blocks (no depth confound vs Foreground-MaskLAM).
        self.assertEqual(self.model.interaction.f_a.num_res_blocks, 2)

    def test_action_blind_at_init(self):
        # Zero-init write-back => interaction contributes nothing => output is independent of z at init.
        y1 = self.model(self.x, self.z, self.m_a, self.m_o)
        y2 = self.model(self.x, torch.randn(self.b, 8), self.m_a, self.m_o)
        self.assertTrue(torch.equal(y1, y2))

    def test_mask_independent_at_init(self):
        # Masks only gate the (zero) write-back at init, so they do not affect the output yet.
        y1 = self.model(self.x, self.z, self.m_a, self.m_o)
        y2 = self.model(self.x, self.z, torch.zeros_like(self.m_a), torch.zeros_like(self.m_o))
        self.assertTrue(torch.equal(y1, y2))

    def test_orthogonal_init_preserves_action_blind_identity(self):
        # use_orthogonal_init re-inits every Conv2d (incl. the write-back projections); the model must
        # re-zero them so the action-blind identity survives.
        model = _model(use_orthogonal_init=True)
        y1 = model(self.x, self.z, self.m_a, self.m_o)
        y2 = model(self.x, torch.randn(self.b, 8), self.m_a, self.m_o)
        self.assertTrue(torch.equal(y1, y2))

    def test_condition_receives_gradient_once_write_back_open(self):
        # Once the write-back opens (Proj != 0), z reaches the output and must carry gradient
        # (the z -> A_hat -> write-back -> decode path).
        with torch.no_grad():
            self.model.interaction.proj_a.weight.normal_(std=0.1)
        z = self.z.clone().requires_grad_(True)
        self.model(self.x, z, self.m_a, self.m_o).sum().backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(z.grad.abs().sum().item(), 0.0)

    def test_return_attn_contract(self):
        out = self.model(self.x, self.z, self.m_a, self.m_o)
        self.assertIsInstance(out, torch.Tensor)
        y, attn = self.model(self.x, self.z, self.m_a, self.m_o, return_attn=True)
        self.assertEqual(tuple(attn.shape), (self.b, 4, 16, 32))  # (B, heads, N, 2N)

    def test_non_square_bottleneck_raises(self):
        # The interaction module needs a square bottleneck; a non-square input would break it.
        with self.assertRaises((ValueError, AssertionError)):
            _model(shape=(15, 32, 64))


if __name__ == "__main__":
    unittest.main()
