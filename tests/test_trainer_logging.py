"""Tests for SupervisedTrainer._log (prefixed / 0-based-step vs legacy logging).

Run:  conda_env/bin/python tests/test_trainer_logging.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ifo.common.trainer import SupervisedTrainer  # noqa: E402

_UNSET = "__step_not_passed__"


class _FakeFabric:
    """Captures log_dict calls; records whether a native step was passed."""

    def __init__(self):
        self.calls = []

    def log_dict(self, metrics, step=_UNSET):
        self.calls.append((dict(metrics), step))


def _make_trainer(log_prefix):
    return SupervisedTrainer(fabric=_FakeFabric(), max_epochs=1, log_prefix=log_prefix)


class TrainerLoggingTest(unittest.TestCase):
    def test_prefixed_mode_prefixes_keys_adds_stage_step_and_omits_native_step(self):
        t = _make_trainer("stage_1")
        t.global_step = 5
        t._log({"train/loss": 2.0})
        metrics, step = t.fabric.calls[-1]
        self.assertEqual(metrics, {"stage_1/train/loss": 2.0, "stage_1/step": 5})
        self.assertEqual(step, _UNSET)  # native W&B step must NOT be forced in prefixed mode

    def test_legacy_mode_unprefixed_with_native_step(self):
        t = _make_trainer(None)
        t.global_step = 7
        t._log({"train/loss": 2.0})
        metrics, step = t.fabric.calls[-1]
        self.assertEqual(metrics, {"train/loss": 2.0})
        self.assertEqual(step, 7)  # legacy behavior unchanged

    def test_prefixed_step_tracks_global_step(self):
        t = _make_trainer("stage_2")
        t.global_step = 0
        t._log({"val/loss": 1.0})
        t.global_step = 3
        t._log({"val/loss": 0.5})
        self.assertEqual(t.fabric.calls[0][0], {"stage_2/val/loss": 1.0, "stage_2/step": 0})
        self.assertEqual(t.fabric.calls[1][0], {"stage_2/val/loss": 0.5, "stage_2/step": 3})


if __name__ == "__main__":
    unittest.main(verbosity=2)
