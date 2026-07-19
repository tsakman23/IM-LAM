"""Tests for the IM-LAM dual (separately-normalized agent/object) reconstruction loss.

Run:  conda_env/bin/python tests/test_slapo_dual_loss.py
"""
import os
import sys
import unittest
from unittest.mock import PropertyMock, patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ifo.modules.slapo.module import SLAPOIDMModule  # noqa: E402
from ifo.modules.slapo.utils import area_normalized_masked_mse, dual_masked_reconstruction_loss  # noqa: E402


class AreaNormalizedMaskedMSETest(unittest.TestCase):
    def test_full_mask_equals_plain_mean(self):
        error = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.ones_like(error)
        result = area_normalized_masked_mse(error, mask)
        self.assertAlmostEqual(result.item(), error.mean().item(), places=6)

    def test_partial_mask_averages_only_covered_pixels(self):
        error = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
        mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])  # covers the two pixels worth 1 and 4
        result = area_normalized_masked_mse(error, mask)
        self.assertAlmostEqual(result.item(), (1.0 + 4.0) / 2.0, places=6)

    def test_small_mask_area_does_not_dilute_the_average(self):
        # A mask covering a single high-error pixel should report that pixel's
        # full error, not have it diluted by the rest of the (unmasked) tensor.
        error = torch.tensor([[0.0, 0.0], [0.0, 16.0]])
        mask = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        result = area_normalized_masked_mse(error, mask)
        self.assertAlmostEqual(result.item(), 16.0, places=6)

    def test_empty_mask_yields_zero_not_nan(self):
        # Denominator floor: an all-empty mask (object fully occluded in every
        # sampled target frame) must contribute zero loss, not 0/0 = NaN.
        error = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
        mask = torch.zeros_like(error)
        result = area_normalized_masked_mse(error, mask)
        self.assertTrue(torch.isfinite(result))
        self.assertAlmostEqual(result.item(), 0.0, places=6)

    def test_empty_object_mask_dual_loss_stays_finite_and_backprops(self):
        # Same guard end-to-end through the dual loss: L_O collapses to 0,
        # L = L_A, and the value is differentiable.
        error = torch.tensor([[1.0, 4.0], [9.0, 16.0]], requires_grad=True)
        agent_mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        empty_object_mask = torch.zeros(2, 2)

        loss, loss_agent, loss_object = dual_masked_reconstruction_loss(
            error, agent_mask, empty_object_mask, lambda_o=1.0
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss_object.item(), 0.0, places=6)
        self.assertAlmostEqual(loss.item(), loss_agent.item(), places=6)
        loss.backward()  # must not raise or produce NaN grads
        self.assertTrue(torch.isfinite(error.grad).all())


class DualMaskedReconstructionLossTest(unittest.TestCase):
    def test_matches_hand_computed_terms(self):
        error = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
        agent_mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])   # covers errors 1, 4 -> L_A = 2.5
        object_mask = torch.tensor([[0.0, 0.0], [0.0, 1.0]])  # covers error 16    -> L_O = 16.0
        lambda_o = 0.1

        loss, loss_agent, loss_object = dual_masked_reconstruction_loss(error, agent_mask, object_mask, lambda_o)

        self.assertAlmostEqual(loss_agent.item(), 2.5, places=6)
        self.assertAlmostEqual(loss_object.item(), 16.0, places=6)
        self.assertAlmostEqual(loss.item(), 2.5 + 0.1 * 16.0, places=6)

    def test_lambda_o_zero_drops_object_term(self):
        error = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
        agent_mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        object_mask = torch.tensor([[0.0, 0.0], [0.0, 1.0]])

        loss, loss_agent, _ = dual_masked_reconstruction_loss(error, agent_mask, object_mask, lambda_o=0.0)
        self.assertAlmostEqual(loss.item(), loss_agent.item(), places=6)

    def test_object_weight_is_independent_of_object_area(self):
        # The whole point of the dual loss vs. the union loss: shrinking the
        # object mask's area (with the same per-pixel error inside it) must NOT
        # shrink its contribution, because it has its own normalization.
        error = torch.full((4, 4), 16.0)
        agent_mask = torch.zeros(4, 4)
        agent_mask[0, 0] = 1.0  # 1 agent pixel, irrelevant to the object term

        small_object_mask = torch.zeros(4, 4)
        small_object_mask[3, 3] = 1.0  # 1 object pixel

        large_object_mask = torch.zeros(4, 4)
        large_object_mask[1:3, 1:3] = 1.0  # 4 object pixels, same per-pixel error

        _, _, loss_object_small = dual_masked_reconstruction_loss(error, agent_mask, small_object_mask, lambda_o=1.0)
        _, _, loss_object_large = dual_masked_reconstruction_loss(error, agent_mask, large_object_mask, lambda_o=1.0)

        self.assertAlmostEqual(loss_object_small.item(), loss_object_large.item(), places=6)
        self.assertAlmostEqual(loss_object_small.item(), 16.0, places=6)


# ──────────────────────────────────────────────────────────────────────────
# Integration test: exercise the real SLAPOIDMModule._forward wiring.
# ──────────────────────────────────────────────────────────────────────────

FRAME_STACK = 2
ACTION_DIM = 2
IMG_SHAPE = (1, 2, 2)  # C, H, W


class _FakeWorldModelNet(nn.Module):
    """Minimal stand-in for SLAPOIDM: always predicts an all-zero next observation.

    With a zero prediction, the elementwise squared error equals the target
    observation squared, which keeps the expected loss values easy to hand-verify
    while still exercising the module's real masking/branching logic end-to-end.
    """

    def __init__(self, frame_stack: int = FRAME_STACK):
        super().__init__()
        self.frame_stack = frame_stack
        self.segmentation_net = None

    def forward(self, observation, mask):
        b = observation.shape[0]
        next_observation = torch.zeros(b, *IMG_SHAPE)
        action_distribution = Normal(torch.zeros(b, ACTION_DIM), torch.ones(b, ACTION_DIM))
        vq_loss = torch.tensor(0.0)
        perplexity = torch.tensor(0.0)
        return next_observation, action_distribution, vq_loss, perplexity


def _make_module(**module_kwargs) -> SLAPOIDMModule:
    return SLAPOIDMModule(
        net=_FakeWorldModelNet(),
        batch_size=1,
        optimizer=OmegaConf.create({}),
        mask_observation=False,
        mask_loss=True,
        **module_kwargs,
    )


def _make_batch(agent_frame, object_frame, num_frames: int = 4) -> TensorDict:
    """Build a batch with the given (H, W) agent/object mask values at frame_stack."""
    b = 1
    observation = torch.zeros(b, num_frames, *IMG_SHAPE)
    observation[0, FRAME_STACK, 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    agent_mask = torch.zeros(b, num_frames, 1, 2, 2)
    agent_mask[0, FRAME_STACK, 0] = torch.tensor(agent_frame)

    object_mask = torch.zeros(b, num_frames, 1, 2, 2)
    object_mask[0, FRAME_STACK, 0] = torch.tensor(object_frame)

    action = torch.zeros(b, num_frames, ACTION_DIM)

    return TensorDict({
        "observation": observation,
        "mask": agent_mask,
        "object_mask": object_mask,
        "action": action,
    }, batch_size=[b])


class SLAPOIDMModuleDualLossWiringTest(unittest.TestCase):
    def test_dual_loss_reports_separately_normalized_terms(self):
        module = _make_module(object_dual_loss=True, object_loss_weight=0.1)
        # agent_mask covers target pixels 1, 2 -> L_A = (1 + 4) / 2 = 2.5
        # object_mask covers target pixel 4    -> L_O = 16 / 1 = 16.0
        batch = _make_batch(agent_frame=[[1, 1], [0, 0]], object_frame=[[0, 0], [0, 1]])

        step_dict = module._forward(batch, batch_idx=0, batch_len=2, prefix="train")

        self.assertAlmostEqual(step_dict["reconstruction_loss_agent"].item(), 2.5, places=6)
        self.assertAlmostEqual(step_dict["reconstruction_loss_object"].item(), 16.0, places=6)
        self.assertAlmostEqual(step_dict["reconstruction_loss"].item(), 2.5 + 0.1 * 16.0, places=6)

    def test_dual_loss_and_union_loss_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            _make_module(object_dual_loss=True, object_mask_loss=True)

    def test_dual_loss_requires_object_mask_in_batch(self):
        module = _make_module(object_dual_loss=True)
        batch = _make_batch(agent_frame=[[1, 1], [0, 0]], object_frame=[[0, 0], [0, 1]])
        del batch["object_mask"]
        with self.assertRaises(ValueError):
            module._forward(batch, batch_idx=0, batch_len=2, prefix="train")

    def test_default_flags_leave_plain_masked_loss_unchanged(self):
        module = _make_module()  # object_dual_loss=False, object_mask_loss=False
        batch = _make_batch(agent_frame=[[1, 1], [0, 0]], object_frame=[[0, 0], [0, 1]])

        step_dict = module._forward(batch, batch_idx=0, batch_len=2, prefix="train")

        # Agent-only masked loss, as before: covers pixels worth 1 and 4 -> mean 2.5.
        self.assertAlmostEqual(step_dict["reconstruction_loss"].item(), 2.5, places=6)
        self.assertNotIn("reconstruction_loss_agent", step_dict.keys())


# ──────────────────────────────────────────────────────────────────────────
# log_dual_loss_grad_every: per-term gradient norm diagnostic.
# ──────────────────────────────────────────────────────────────────────────


class _Decoder(nn.Module):
    """A single learnable scalar, standing in for the FDM decoder's parameters."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))


class _FakeWorldModelNetWithDecoder(nn.Module):
    """Like _FakeWorldModelNet, but with a real ``decoder`` submodule so gradients
    actually flow through it and per-term grad norms are non-degenerate."""

    def __init__(self, frame_stack: int = FRAME_STACK):
        super().__init__()
        self.frame_stack = frame_stack
        self.segmentation_net = None
        self.decoder = _Decoder()

    def forward(self, observation, mask):
        b = observation.shape[0]
        next_observation = self.decoder.scale * torch.ones(b, *IMG_SHAPE)
        action_distribution = Normal(torch.zeros(b, ACTION_DIM), torch.ones(b, ACTION_DIM))
        vq_loss = torch.tensor(0.0)
        perplexity = torch.tensor(0.0)
        return next_observation, action_distribution, vq_loss, perplexity


class SLAPOIDMModuleGradDiagnosticsTest(unittest.TestCase):
    def _make_module_with_decoder(self, **module_kwargs) -> SLAPOIDMModule:
        return SLAPOIDMModule(
            net=_FakeWorldModelNetWithDecoder(),
            batch_size=1,
            optimizer=OmegaConf.create({}),
            mask_observation=False,
            mask_loss=True,
            object_dual_loss=True,
            object_loss_weight=1.0,
            **module_kwargs,
        )

    def test_disabled_by_default_no_extra_keys(self):
        module = self._make_module_with_decoder()  # log_dual_loss_grad_every=0
        batch = _make_batch(agent_frame=[[1, 1], [0, 0]], object_frame=[[0, 0], [0, 1]])

        step_dict = module._forward(batch, batch_idx=0, batch_len=2, prefix="train")

        self.assertNotIn("reconstruction_loss_object_grad_norm", step_dict.keys())

    def test_grad_norms_match_hand_computed_values(self):
        # decoder predicts a constant `scale` (init 1.0) at every pixel; target at
        # frame_stack is [[1, 2], [3, 4]]. agent_mask covers targets 1, 2; object_mask
        # covers target 4 (same layout as the other wiring tests above).
        #   d(loss_agent)/d(scale)  = mean(2*(1-1), 2*(1-2))     = -1.0  -> |g_A| = 1.0
        #   d(loss_object)/d(scale) = 2*(1-4)                     = -6.0  -> |g_O| = 6.0
        module = self._make_module_with_decoder(log_dual_loss_grad_every=1)
        batch = _make_batch(agent_frame=[[1, 1], [0, 0]], object_frame=[[0, 0], [0, 1]])

        step_dict = module._forward(batch, batch_idx=0, batch_len=2, prefix="train")

        self.assertAlmostEqual(step_dict["reconstruction_loss_agent_grad_norm"].item(), 1.0, places=5)
        self.assertAlmostEqual(step_dict["reconstruction_loss_object_grad_norm"].item(), 6.0, places=5)
        self.assertNotIn("reconstruction_loss_object_grad_share", step_dict.keys())

    def test_diagnostic_does_not_break_the_real_backward_pass(self):
        # retain_graph=True in the diagnostic must leave enough of the graph alive
        # for the *actual* optimization backward on the combined loss afterwards.
        module = self._make_module_with_decoder(log_dual_loss_grad_every=1)
        batch = _make_batch(agent_frame=[[1, 1], [0, 0]], object_frame=[[0, 0], [0, 1]])

        step_dict = module._forward(batch, batch_idx=0, batch_len=2, prefix="train")
        step_dict["loss"].backward()  # must not raise

        self.assertIsNotNone(module.net.decoder.scale.grad)
        # loss = L_A + 1.0 * L_O (object_loss_weight=1.0), so d(loss)/d(scale) = g_A + g_O
        # with sign: d(L_A)/d(scale) = -1.0, d(L_O)/d(scale) = -6.0 -> total -7.0
        self.assertAlmostEqual(module.net.decoder.scale.grad.item(), -7.0, places=5)

    def test_respects_logging_interval(self):
        module = self._make_module_with_decoder(log_dual_loss_grad_every=10)
        batch = _make_batch(agent_frame=[[1, 1], [0, 0]], object_frame=[[0, 0], [0, 1]])

        with patch.object(type(module), "global_step", new_callable=PropertyMock) as mock_step:
            mock_step.return_value = 23  # 23 % 10 != 0 -> should NOT log this step
            step_dict_skip = module._forward(batch, batch_idx=0, batch_len=2, prefix="train")
            self.assertNotIn("reconstruction_loss_object_grad_norm", step_dict_skip.keys())

            mock_step.return_value = 30  # 30 % 10 == 0 -> should log this step
            step_dict_log = module._forward(batch, batch_idx=0, batch_len=2, prefix="train")
            self.assertIn("reconstruction_loss_object_grad_norm", step_dict_log.keys())

    def test_validation_prefix_skips_the_diagnostic(self):
        # prefix="val" must never attempt autograd.grad (validation runs under
        # inference/no-grad mode in the real Lightning trainer).
        module = self._make_module_with_decoder(log_dual_loss_grad_every=1)
        batch = _make_batch(agent_frame=[[1, 1], [0, 0]], object_frame=[[0, 0], [0, 1]])

        step_dict = module._forward(batch, batch_idx=0, batch_len=2, prefix="val")

        self.assertNotIn("reconstruction_loss_object_grad_norm", step_dict.keys())


if __name__ == "__main__":
    unittest.main(verbosity=2)
