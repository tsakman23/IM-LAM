"""Tests for the IM-LAM Stage-1 Hydra configs (Phase 6).

Run:  conda_env/bin/python tests/test_imlam_configs.py

Each config is composed and its net instantiated with a synthetic observation/action space (the
same call the experiment makes: instantiate(cfg.net)(observation_space=..., action_space=...)), so a
broken _target_, a bad num_heads, or a missing kwarg fails here rather than seven hours into a run. A
small 32x32 space keeps the DMW world_model params (192-wide bottleneck, num_heads=6) valid but fast.
"""
import os
import sys
import unittest

import numpy as np
import hydra
from gymnasium import spaces
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ifo.common.nets.interaction_world_model import InteractionWorldModel
from ifo.modules.slapo.callbacks import ObjectLossWeightAnneal
from ifo.modules.slapo.nets import IMLAMIDM

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "configs"))


def _instantiate_net(config_name, overrides=None):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    obs = spaces.Box(low=0.0, high=1.0, shape=(13, 3, 32, 32), dtype=np.float32)  # t = offset(10)+stack(3)
    act = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    net = hydra.utils.instantiate(cfg.net)(observation_space=obs, action_space=act)
    return cfg, net


class IMLAMConfigTest(unittest.TestCase):
    def test_union_config_builds_imlam_with_interaction_fdm(self):
        cfg, net = _instantiate_net("imlam_dmw_stage_1")
        self.assertIsInstance(net, IMLAMIDM)
        self.assertIsInstance(net.decoder, InteractionWorldModel)
        # Union loss over agent U object; IDM mask input stays agent-only.
        self.assertTrue(cfg.module.object_mask_loss)
        self.assertFalse(cfg.module.object_mask_input)
        self.assertFalse(cfg.module.get("object_dual_loss", False))
        # Object mask + state must be loaded (state for the Phase-8 object-dynamics probe).
        self.assertTrue(cfg.dataset.with_object_mask)
        self.assertTrue(cfg.dataset.with_object_state)
        # Checkpoint contract preserved: the IDM encoder is byte-identical to MaskLAM's (32 keys).
        self.assertEqual(len([k for k in net.state_dict() if k.startswith("encoder.")]), 32)
        # num_heads must divide the 192-wide bottleneck (192 / 6 = 32).
        self.assertEqual(cfg.net.world_model.num_heads, 6)

    def test_dual_config_enables_mutually_exclusive_dual_loss_and_grad_diag(self):
        cfg, net = _instantiate_net("imlam_dual_dmw_stage_1")
        self.assertIsInstance(net, IMLAMIDM)
        self.assertFalse(cfg.module.object_mask_loss)  # mutually exclusive with dual
        self.assertTrue(cfg.module.object_dual_loss)
        self.assertEqual(cfg.module.object_loss_weight, 1.0)
        self.assertGreater(cfg.module.log_dual_loss_grad_every, 0)  # L_O grad diagnostic enabled

    def test_dual_anneal_config_wires_the_object_weight_warmup(self):
        cfg, net = _instantiate_net("imlam_dual_anneal_dmw_stage_1")
        self.assertIsInstance(net, IMLAMIDM)
        # Inherits the dual loss...
        self.assertTrue(cfg.module.object_dual_loss)
        # ...but initializes the weight at the anneal start (0.0), not the fixed 1.0.
        self.assertEqual(cfg.module.object_loss_weight, 0.0)
        # The warm-up block must instantiate to the callback the experiment appends.
        cb = hydra.utils.instantiate(cfg.object_loss_weight_anneal)
        self.assertIsInstance(cb, ObjectLossWeightAnneal)
        self.assertEqual(cfg.object_loss_weight_anneal.start, cfg.module.object_loss_weight)

    def test_direct_z_ablation_flag_reaches_the_object_branch(self):
        cfg, net = _instantiate_net("imlam_direct_z_dmw_stage_1")
        self.assertIsInstance(net, IMLAMIDM)
        self.assertTrue(cfg.net.world_model.direct_z_to_object)
        # The flag must actually reach the interaction module, not just sit in the config.
        self.assertTrue(net.decoder.interaction.direct_z_to_object)
        # It inherits the union setup (object mask loss on).
        self.assertTrue(cfg.module.object_mask_loss)
        # Default task must be a large-object task: on push (~0.2 bottleneck cells) the object branch is
        # nearly inert, so the matched ablation would tie uninformatively (proposal S6.2).
        self.assertNotIn("push", cfg.env.name)


if __name__ == "__main__":
    unittest.main()
