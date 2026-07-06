"""Tests for SupervisedTrainer._log (in-process pipeline vs legacy logging).

Run:  conda_env/bin/python tests/test_trainer_logging.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ifo.common.trainer import SupervisedTrainer  # noqa: E402

_UNSET = "__step_not_passed__"


class _FakeFabric:
    """Captures fabric.log_dict(metrics, step) calls; optional cumulative step offset."""

    def __init__(self, step_offset=None):
        self.calls = []
        if step_offset is not None:
            self._pipeline_step_offset = step_offset

    def log_dict(self, metrics, step=_UNSET):
        self.calls.append((dict(metrics), step))


def _make_trainer(log_prefix, step_offset=None):
    t = SupervisedTrainer(fabric=_FakeFabric(step_offset=step_offset), max_epochs=1, log_prefix=log_prefix)
    return t


class TrainerLoggingTest(unittest.TestCase):
    def test_pipeline_mode_prefixes_keys_and_uses_global_step_when_offset_zero(self):
        t = _make_trainer("stage_1")  # no offset set -> treated as 0
        t.global_step = 5
        t._log({"train/loss": 2.0})
        metrics, step = t.fabric.calls[-1]
        self.assertEqual(metrics, {"stage_1/train/loss": 2.0})
        self.assertEqual(step, 5)

    def test_pipeline_mode_adds_cumulative_offset_to_step(self):
        t = _make_trainer("stage_2", step_offset=100)  # a prior stage logged 100 steps
        t.global_step = 5
        t._log({"val/loss": 0.5})
        metrics, step = t.fabric.calls[-1]
        self.assertEqual(metrics, {"stage_2/val/loss": 0.5})
        self.assertEqual(step, 105)  # monotonic, cumulative across stages

    def test_legacy_mode_unprefixed_with_native_step(self):
        t = _make_trainer(None)
        t.global_step = 7
        t._log({"train/loss": 2.0})
        metrics, step = t.fabric.calls[-1]
        self.assertEqual(metrics, {"train/loss": 2.0})
        self.assertEqual(step, 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
