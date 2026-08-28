"""Tests for ObjectLossWeightAnneal (dual-loss object-term warm-up).

Run:  conda_env/bin/python tests/test_object_loss_anneal.py

The callback ramps the module's object_loss_weight BUFFER from `start` to `end` over `anneal_steps`,
after an initial `hold_steps` window at `start`. Warming the object term up (rather than applying full
lambda_o from step 0) lets the agent pathway and the latent z settle before the dual loss's
amplified small-object gradient hits - the IM-LAM x dual instability. Pure-logic tests: a namespace
stands in for the module (only needs an `object_loss_weight` tensor) and for the trainer (only
`global_step`), mirroring how the real callback reads them.
"""
import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ifo.modules.slapo.callbacks import ObjectLossWeightAnneal


def _model(w=1.0):
    return types.SimpleNamespace(object_loss_weight=torch.tensor(float(w)))


def _trainer(step):
    return types.SimpleNamespace(global_step=step)


class ObjectLossWeightAnnealTest(unittest.TestCase):
    def test_holds_at_start_through_the_hold_window(self):
        cb = ObjectLossWeightAnneal(start=0.0, end=1.0, anneal_steps=100, hold_steps=50)
        m = _model(w=1.0)
        cb.train_batch_start(_trainer(0), m)
        self.assertAlmostEqual(m.object_loss_weight.item(), 0.0, places=6)
        cb.train_batch_start(_trainer(49), m)
        self.assertAlmostEqual(m.object_loss_weight.item(), 0.0, places=6)

    def test_ramps_linearly_after_the_hold(self):
        cb = ObjectLossWeightAnneal(start=0.0, end=1.0, anneal_steps=100, hold_steps=50)
        m = _model()
        cb.train_batch_start(_trainer(100), m)  # 50 steps into a 100-step ramp -> halfway
        self.assertAlmostEqual(m.object_loss_weight.item(), 0.5, places=6)

    def test_holds_at_end_after_the_ramp(self):
        cb = ObjectLossWeightAnneal(start=0.0, end=1.0, anneal_steps=100, hold_steps=50)
        m = _model()
        cb.train_batch_start(_trainer(10_000), m)
        self.assertAlmostEqual(m.object_loss_weight.item(), 1.0, places=6)

    def test_default_no_hold_ramps_from_step_zero(self):
        cb = ObjectLossWeightAnneal(start=0.2, end=1.0, anneal_steps=100)  # hold_steps defaults to 0
        m = _model()
        cb.train_batch_start(_trainer(0), m)
        self.assertAlmostEqual(m.object_loss_weight.item(), 0.2, places=6)
        cb.train_batch_start(_trainer(50), m)
        self.assertAlmostEqual(m.object_loss_weight.item(), 0.2 + 0.8 * 0.5, places=6)

    def test_missing_buffer_is_a_noop(self):
        cb = ObjectLossWeightAnneal(start=0.0, end=1.0, anneal_steps=100)
        cb.train_batch_start(_trainer(10), types.SimpleNamespace())  # must not raise

    def test_non_tensor_weight_is_left_untouched(self):
        cb = ObjectLossWeightAnneal(start=0.0, end=1.0, anneal_steps=100)
        m = types.SimpleNamespace(object_loss_weight=1.0)  # a plain float, not a buffer
        cb.train_batch_start(_trainer(10), m)  # must not raise
        self.assertEqual(m.object_loss_weight, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
