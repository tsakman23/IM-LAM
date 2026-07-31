"""Tests for the IM-LAM module wiring (Phase 5).

Run:  conda_env/bin/python tests/test_imlam_module_wiring.py

Phase 5 makes two backward-compatible edits to SLAPOIDMModule._forward:
  1. it passes the batch's object_mask to the net (safe for all nets: the parent SLAPOIDM.forward
     accepts and ignores object_mask), so IMLAMIDM's FDM can consume it;
  2. it surfaces the interaction FDM's health scalars (per-head beta, write-back proj norms) into
     step_dict when the net exposes interaction_diagnostics(), and leaves plain MaskLAM/Foreground
     runs (no such method) untouched.

These tests exercise the real module with fake nets, mirroring tests/test_slapo_dual_loss.py.
"""
import os
import sys
import unittest

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ifo.modules.slapo.module import SLAPOIDMModule  # noqa: E402

FRAME_STACK = 2
ACTION_DIM = 2
IMG_SHAPE = (1, 2, 2)  # C, H, W


class _FakeIMLAMNet(nn.Module):
    """Stand-in for IMLAMIDM: records the object_mask it was given and exposes diagnostics."""

    def __init__(self, frame_stack: int = FRAME_STACK):
        super().__init__()
        self.frame_stack = frame_stack
        self.segmentation_net = None
        self.future_obs_sampling = True  # the val diagnostics should toggle this off and restore it
        self.received_object_mask = "UNSET"
        self.modes_seen = []

    def forward(self, observation, mask, object_mask=None, agent_ctx_mode="normal", return_attn=False):
        self.received_object_mask = object_mask
        self.modes_seen.append(agent_ctx_mode)
        b = observation.shape[0]
        # Mode-dependent prediction so the agent-path ratios are non-degenerate.
        fill = {"normal": 0.0, "no_transition": 0.1, "shuffled": 0.2}[agent_ctx_mode]
        next_observation = torch.full((b, *IMG_SHAPE), fill)
        action_distribution = Normal(torch.zeros(b, ACTION_DIM), torch.ones(b, ACTION_DIM))
        return next_observation, action_distribution, torch.tensor(0.0), torch.tensor(0.0)

    def interaction_diagnostics(self):
        return {
            "beta_msa_a": torch.tensor(2.0),
            "beta_msa_o": torch.tensor(2.0),
            "proj_a_norm": torch.tensor(0.0),
            "proj_o_norm": torch.tensor(0.5),
        }

    def extract_entities(self, observation, mask, object_mask):
        # IM-LAM marker + probe hook; dummy (A_t, O_t) token sets.
        b = observation.shape[0]
        return torch.zeros(b, 4, 8), torch.ones(b, 4, 8)


class _FakePlainNet(nn.Module):
    """Stand-in for MaskLAM/Foreground: accepts object_mask (ignored), no interaction diagnostics."""

    def __init__(self, frame_stack: int = FRAME_STACK, predict_nan: bool = False):
        super().__init__()
        self.frame_stack = frame_stack
        self.segmentation_net = None
        self.predict_nan = predict_nan

    def forward(self, observation, mask, object_mask=None):
        b = observation.shape[0]
        fill = float("nan") if self.predict_nan else 0.0
        next_observation = torch.full((b, *IMG_SHAPE), fill)
        action_distribution = Normal(torch.zeros(b, ACTION_DIM), torch.ones(b, ACTION_DIM))
        return next_observation, action_distribution, torch.tensor(0.0), torch.tensor(0.0)


def _make_module(net, **module_kwargs) -> SLAPOIDMModule:
    return SLAPOIDMModule(
        net=net,
        batch_size=1,
        optimizer=OmegaConf.create({}),
        mask_observation=False,
        mask_loss=True,
        **module_kwargs,
    )


def _make_batch(with_object_mask: bool = True, num_frames: int = 4) -> TensorDict:
    b = 1
    observation = torch.zeros(b, num_frames, *IMG_SHAPE)
    observation[0, FRAME_STACK, 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    agent_mask = torch.zeros(b, num_frames, 1, 2, 2)
    agent_mask[0, FRAME_STACK, 0] = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    data = {"observation": observation, "mask": agent_mask, "action": torch.zeros(b, num_frames, ACTION_DIM)}
    if with_object_mask:
        object_mask = torch.zeros(b, num_frames, 1, 2, 2)
        object_mask[0, FRAME_STACK, 0] = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        data["object_mask"] = object_mask
    return TensorDict(data, batch_size=[b])


class IMLAMModuleWiringTest(unittest.TestCase):
    def test_module_passes_object_mask_to_the_net(self):
        net = _FakeIMLAMNet()
        module = _make_module(net, object_mask_loss=True)
        batch = _make_batch()
        module._forward(batch, batch_idx=0, batch_len=2, prefix="train")
        self.assertIsNotNone(net.received_object_mask)
        self.assertTrue(torch.equal(net.received_object_mask, batch["object_mask"]))

    def test_union_loss_is_finite(self):
        module = _make_module(_FakeIMLAMNet(), object_mask_loss=True)
        step_dict = module._forward(_make_batch(), batch_idx=0, batch_len=2, prefix="train")
        self.assertTrue(torch.isfinite(step_dict["reconstruction_loss"]))

    def test_dual_loss_is_finite_and_reports_terms(self):
        module = _make_module(_FakeIMLAMNet(), object_dual_loss=True, object_loss_weight=1.0)
        step_dict = module._forward(_make_batch(), batch_idx=0, batch_len=2, prefix="train")
        self.assertTrue(torch.isfinite(step_dict["reconstruction_loss"]))
        self.assertIn("reconstruction_loss_agent", step_dict.keys())
        self.assertIn("reconstruction_loss_object", step_dict.keys())

    def test_interaction_diagnostics_surface_in_step_dict(self):
        module = _make_module(_FakeIMLAMNet(), object_mask_loss=True)
        step_dict = module._forward(_make_batch(), batch_idx=0, batch_len=2, prefix="train")
        for key in ("beta_msa_a", "beta_msa_o", "proj_a_norm", "proj_o_norm"):
            self.assertIn(key, step_dict.keys())
        self.assertAlmostEqual(step_dict["proj_o_norm"].item(), 0.5, places=6)

    def test_loss_nonfinite_flag_detects_blowups(self):
        # A NaN/Inf loss (max_grad_norm is null on DMW, so a blow-up would otherwise be unlogged) must
        # be surfaced as a step_dict flag; a healthy step reports 0.
        healthy = _make_module(_FakePlainNet())._forward(
            _make_batch(with_object_mask=False), batch_idx=0, batch_len=2, prefix="train"
        )
        self.assertEqual(healthy["loss_nonfinite"].item(), 0.0)
        blown = _make_module(_FakePlainNet(predict_nan=True))._forward(
            _make_batch(with_object_mask=False), batch_idx=0, batch_len=2, prefix="train"
        )
        self.assertFalse(torch.isfinite(blown["loss"]))
        self.assertEqual(blown["loss_nonfinite"].item(), 1.0)

    def test_validation_logs_object_diagnostics_for_imlam(self):
        # In validation, E_O / copy / ratio + the IM-LAM agent-path ratio must appear (the trainer means
        # every val step_dict key under val/, so they log to W&B per run with no standalone script).
        net = _FakeIMLAMNet()
        step_dict = _make_module(net, object_mask_loss=True)._forward(
            _make_batch(), batch_idx=0, batch_len=2, prefix="val"
        )
        for key in ("object_E_O", "object_E_O_copy", "object_prediction_ratio", "object_R_no_transition"):
            self.assertIn(key, step_dict.keys())
        self.assertIn("no_transition", net.modes_seen)     # agent_ctx_mode reached the net
        self.assertTrue(net.future_obs_sampling)           # toggled off during the diagnostic, then restored

    def test_training_does_not_log_object_diagnostics(self):
        # The object diagnostics are eval-only; training steps must not pay the extra forwards.
        step_dict = _make_module(_FakeIMLAMNet(), object_mask_loss=True)._forward(
            _make_batch(), batch_idx=0, batch_len=2, prefix="train"
        )
        self.assertNotIn("object_E_O", step_dict.keys())

    def test_baseline_val_logs_E_O_but_no_agent_path(self):
        # E_O works for any net with an object mask (Foreground); agent-path is IM-LAM-only (no
        # extract_entities -> no predicted agent transition to corrupt).
        step_dict = _make_module(_FakePlainNet(), object_mask_loss=True)._forward(
            _make_batch(), batch_idx=0, batch_len=2, prefix="val"
        )
        self.assertIn("object_E_O", step_dict.keys())
        self.assertNotIn("object_R_no_transition", step_dict.keys())

    def test_plain_net_gets_no_interaction_diagnostics(self):
        # A MaskLAM/Foreground net (no interaction_diagnostics) must not gain instrumentation keys and
        # must run unchanged when no object_mask is present in the batch.
        module = _make_module(_FakePlainNet())  # no object flags
        step_dict = module._forward(_make_batch(with_object_mask=False), batch_idx=0, batch_len=2, prefix="train")
        self.assertTrue(torch.isfinite(step_dict["reconstruction_loss"]))
        for key in ("beta_msa_a", "beta_msa_o", "proj_a_norm", "proj_o_norm"):
            self.assertNotIn(key, step_dict.keys())


if __name__ == "__main__":
    unittest.main()
