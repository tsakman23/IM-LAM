"""Tests for LAPOIDMModule's action_decoder_nmse (mirrors SLAPOIDMModule's existing behavior).

Run:  conda_env/bin/python tests/test_lapo_action_nmse.py

LAPOIDMModule always logs action_decoder_mse (via get_action_metrics), but had no action_variance
constructor arg at all, so it could never compute the normalized action_decoder_nmse = mse/Var(a) that
SLAPOIDMModule does live during training. NMSE for existing LAPO runs only ever existed via a manual
post-hoc W&B backfill (quicktest.py). These tests pin the same action_variance=None-disables /
action_variance=set-enables contract for LAPOIDMModule, so future LAPO runs log it live too.
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
from ifo.modules.lapo.module import LAPOIDMModule  # noqa: E402

ACTION_DIM = 2
IMG_SHAPE = (1, 2, 2)  # C, H, W


class _FakeLAPONet(nn.Module):
    """Stand-in for LAPO's IDM net: single-arg forward on the observation only (no mask)."""

    def __init__(self):
        super().__init__()
        self.segmentation_net = None

    def forward(self, observation):
        b = observation.shape[0]
        next_observation = torch.zeros(b, *IMG_SHAPE)
        action_distribution = Normal(torch.zeros(b, ACTION_DIM), torch.ones(b, ACTION_DIM))
        return next_observation, action_distribution, torch.tensor(0.0), torch.tensor(0.0)


def _make_module(**kwargs) -> LAPOIDMModule:
    return LAPOIDMModule(
        net=_FakeLAPONet(),
        batch_size=1,
        optimizer=OmegaConf.create({}),
        **kwargs,
    )


def _make_batch(num_frames: int = 2) -> TensorDict:
    b = 1
    return TensorDict({
        "observation": torch.zeros(b, num_frames, *IMG_SHAPE),
        "action": torch.zeros(b, num_frames, ACTION_DIM),
    }, batch_size=[b])


class LAPOActionNMSETest(unittest.TestCase):
    def test_action_variance_none_disables_nmse(self):
        # Default (no action_variance passed): action_decoder_mse is logged, action_decoder_nmse is not -
        # matches SLAPOIDMModule's "None disables the metric" contract, not an error.
        module = _make_module()
        step_dict = module._forward(_make_batch(), batch_idx=0, batch_len=1, prefix="train")
        self.assertIn("action_decoder_mse", step_dict.keys())
        self.assertNotIn("action_decoder_nmse", step_dict.keys())

    def test_action_variance_set_computes_nmse(self):
        module = _make_module(action_variance=0.25)
        step_dict = module._forward(_make_batch(), batch_idx=0, batch_len=1, prefix="train")
        self.assertIn("action_decoder_nmse", step_dict.keys())
        expected = step_dict["action_decoder_mse"] / 0.25
        self.assertAlmostEqual(step_dict["action_decoder_nmse"].item(), expected.item(), places=6)


if __name__ == "__main__":
    unittest.main()
