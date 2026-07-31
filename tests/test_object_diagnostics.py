"""Tests for the IM-LAM Stage-1 diagnostic metric cores (Phase 8).

Run:  conda_env/bin/python tests/test_object_diagnostics.py

Covers the pure metrics on synthetic inputs (copy-baseline ratio, linear-probe R²/NMSE, separation
ratio) and one model-level check that the agent_ctx_mode substitution actually changes the held-out
object error - the mechanism the agent-path-dependence diagnostic relies on.
"""
import functools
import os
import sys
import unittest

import numpy as np
import torch
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ifo.common.nets.interaction_world_model import InteractionWorldModel
from ifo.common.nets.quantizer import IdentityQuantizer
from ifo.modules.slapo.nets import IMLAMIDM
from scripts.imlam_diagnostics.metrics import (
    copy_baseline_error,
    fit_linear_probe,
    linear_probe_r2,
    masked_sq_error_and_area,
    object_prediction_error,
    object_prediction_ratio,
    quat_geodesic_angle,
    separation_ratio,
)


class ObjectPredictionErrorTest(unittest.TestCase):
    def test_perfect_prediction_is_zero(self):
        target = torch.rand(2, 3, 4, 4)
        mask = (torch.rand(2, 1, 4, 4) > 0.5).float()
        self.assertAlmostEqual(object_prediction_error(target.clone(), target, mask).item(), 0.0, places=6)

    def test_copy_baseline_and_ratio_hand_computed(self):
        # One object pixel; current differs from target there by 0.3 -> E_O^copy = 0.3^2 = 0.09.
        current = torch.tensor([[[[0.0, 0.0], [0.0, 0.2]]]])
        target = torch.tensor([[[[0.0, 0.0], [0.0, 0.5]]]])
        mask = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]])
        e_copy = copy_baseline_error(current, target, mask)
        self.assertAlmostEqual(e_copy.item(), 0.09, places=6)
        # A model that predicts the target exactly beats copy: E_O=0, ratio ~ 0.
        e_model = object_prediction_error(target.clone(), target, mask)
        self.assertAlmostEqual(object_prediction_ratio(e_model, e_copy).item(), 0.0, places=6)

    def test_ratio_above_one_when_model_worse_than_copy(self):
        e_model = torch.tensor(0.20)
        e_copy = torch.tensor(0.05)
        self.assertGreater(object_prediction_ratio(e_model, e_copy).item(), 1.0)

    def test_pooled_estimator_weights_by_area_not_batch(self):
        # S7.4.1 pooled estimator = sum_error / sum_area, NOT a batch-mean of normalized errors.
        # Batch 1: 1 object pixel, sq error 1 -> E=1. Batch 2: 3 object pixels, sq error 9 each -> E=9.
        # Pooled = (1 + 27) / (1 + 3) = 7.0; a batch-mean would give (1 + 9) / 2 = 5.0.
        b1 = (torch.tensor([[[[1.0, 0.0]]]]), torch.zeros(1, 1, 1, 2), torch.tensor([[[[1.0, 0.0]]]]))
        b2 = (torch.tensor([[[[3.0, 3.0, 3.0, 0.0]]]]), torch.zeros(1, 1, 1, 4),
              torch.tensor([[[[1.0, 1.0, 1.0, 0.0]]]]))
        err_sum = area_sum = 0.0
        for pred, target, mask in (b1, b2):
            e, a = masked_sq_error_and_area(pred, target, mask)
            err_sum, area_sum = err_sum + e, area_sum + a
        self.assertAlmostEqual((err_sum / area_sum).item(), 7.0, places=5)


class QuatGeodesicAngleTest(unittest.TestCase):
    def test_identity_and_sign_invariance(self):
        import math
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.assertAlmostEqual(quat_geodesic_angle(q, q).item(), 0.0, places=5)
        self.assertAlmostEqual(quat_geodesic_angle(q, -q).item(), 0.0, places=5)  # q and -q are one rotation
        q90 = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])  # 90 deg about z
        self.assertAlmostEqual(quat_geodesic_angle(q, q90).item(), math.pi / 2, places=4)


class LinearProbeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.N, self.d, self.k = 200, 8, 3
        self.X = torch.randn(self.N, self.d)

    def test_recovers_linear_map_held_out(self):
        # targets are an exact linear function of the features -> held-out R² ~ 1, NMSE ~ 0.
        w_true, b_true = torch.randn(self.d, self.k), torch.randn(self.k)
        y = self.X @ w_true + b_true
        s = linear_probe_r2(self.X[:150], y[:150], eval_features=self.X[150:], eval_targets=y[150:])
        self.assertGreater(s["r2"].item(), 0.999)
        self.assertLess(s["nmse"].item(), 1e-3)

    def test_low_r2_for_unpredictable_target_held_out(self):
        # targets independent of features -> held-out R² near 0 (no linear signal to exploit).
        y = torch.randn(self.N, self.k)
        s = linear_probe_r2(self.X[:150], y[:150], eval_features=self.X[150:], eval_targets=y[150:])
        self.assertLess(s["r2"].item(), 0.2)

    def test_probe_nmse_uses_target_variance_not_action_variance(self):
        # NMSE is normalized by Var(targets) computed from the data; scaling the targets by c leaves NMSE
        # invariant (both residual and variance scale by c^2), confirming it is not an absolute error.
        y = self.X @ torch.randn(self.d, self.k) + 0.3 * torch.randn(self.N, self.k)
        s1 = linear_probe_r2(self.X, y)
        s2 = linear_probe_r2(self.X, 5.0 * y)
        self.assertAlmostEqual(s1["nmse"].item(), s2["nmse"].item(), places=4)


class SeparationRatioTest(unittest.TestCase):
    def test_low_when_object_readout_is_the_better_predictor(self):
        self.assertLess(separation_ratio(torch.tensor(0.1), torch.tensor(0.9)).item(), 0.2)

    def test_near_one_when_agent_predicts_object_motion_equally(self):
        self.assertAlmostEqual(separation_ratio(torch.tensor(0.8), torch.tensor(0.8)).item(), 1.0, places=3)


FRAME_STACK, OFFSET, C, H, W, ACT_DIM = 3, 2, 3, 32, 32, 4


def _imlam(**kw):
    obs = spaces.Box(0.0, 1.0, (FRAME_STACK + OFFSET, C, H, W), np.float32)
    act = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
    wm = functools.partial(InteractionWorldModel, channel_multiplier=1, encoder_channels=(16, 32, 32),
                           encoder_num_res_blocks=2, num_heads=4)
    return IMLAMIDM(observation_space=obs, action_space=act, channel_multiplier=1, quantizer=IdentityQuantizer(),
                    world_model=wm, code_dim=8, num_latents=1, frame_stack=FRAME_STACK,
                    future_obs_offset=OFFSET, add_mask_to_observation=True, **kw)


class AgentPathDependenceTest(unittest.TestCase):
    def test_agent_ctx_mode_substitution_changes_object_error(self):
        # The agent-path-dependence diagnostic corrupts the predicted agent transition and measures the
        # change in object error. That requires the substitution to actually reach the object prediction.
        torch.manual_seed(0)
        net = _imlam().eval()
        with torch.no_grad():
            net.decoder.interaction.proj_o.weight.normal_(std=0.1)  # open the object write-back
        b = 2
        x = torch.randn(b, FRAME_STACK + OFFSET, C, H, W)
        agent_mask = (torch.rand(b, FRAME_STACK + OFFSET, 1, H, W) > 0.5).float()
        object_mask = (torch.rand(b, FRAME_STACK + OFFSET, 1, H, W) > 0.5).float()
        target = x[:, FRAME_STACK]
        om = object_mask[:, FRAME_STACK]
        with torch.no_grad():
            y_normal = net(x, agent_mask, object_mask=object_mask, agent_ctx_mode="normal")[0]
            y_notrans = net(x, agent_mask, object_mask=object_mask, agent_ctx_mode="no_transition")[0]
        e_normal = object_prediction_error(y_normal, target, om)
        e_notrans = object_prediction_error(y_notrans, target, om)
        # The substitution must propagate to E_O (mechanism check). At init F_A's Â_{t+1} barely differs
        # from A_t so the change is tiny; a trained model amplifies it into the load-bearing R_no-transition.
        self.assertFalse(torch.equal(e_normal, e_notrans))


if __name__ == "__main__":
    unittest.main()
