"""Tests for SupervisedTrainer._log (in-process pipeline vs legacy logging).

Run:  conda_env/bin/python tests/test_trainer_logging.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ifo.common.trainer import SupervisedTrainer  # noqa: E402

_UNSET = "__step_not_passed__"


class _FakeRun:
    """Captures raw wandb-run .log() payloads (in-process pipeline path)."""

    def __init__(self):
        self.logged = []

    def log(self, payload):
        self.logged.append(dict(payload))


class _FakeFabric:
    """Captures fabric.log_dict calls (legacy path); optional `_pipeline_run`."""

    def __init__(self, pipeline_run=None):
        self.calls = []
        if pipeline_run is not None:
            self._pipeline_run = pipeline_run

    def log_dict(self, metrics, step=_UNSET):
        self.calls.append((dict(metrics), step))


def _make_trainer(log_prefix, pipeline_run=None):
    return SupervisedTrainer(
        fabric=_FakeFabric(pipeline_run=pipeline_run), max_epochs=1, log_prefix=log_prefix
    )


class TrainerLoggingTest(unittest.TestCase):
    def test_pipeline_mode_logs_prefixed_keys_and_stage_step_to_raw_run(self):
        run = _FakeRun()
        t = _make_trainer("stage_1", pipeline_run=run)
        t.global_step = 5
        t._log({"train/loss": 2.0})
        self.assertEqual(run.logged[-1], {"stage_1/train/loss": 2.0, "stage_1/step": 5})
        self.assertEqual(t.fabric.calls, [])  # did NOT go through the fabric logger

    def test_pipeline_step_axis_tracks_global_step(self):
        run = _FakeRun()
        t = _make_trainer("stage_2", pipeline_run=run)
        t.global_step = 0
        t._log({"val/loss": 1.0})
        t.global_step = 3
        t._log({"val/loss": 0.5})
        self.assertEqual(run.logged[0], {"stage_2/val/loss": 1.0, "stage_2/step": 0})
        self.assertEqual(run.logged[1], {"stage_2/val/loss": 0.5, "stage_2/step": 3})

    def test_legacy_mode_unprefixed_with_native_step(self):
        t = _make_trainer(None)  # no pipeline run, no prefix
        t.global_step = 7
        t._log({"train/loss": 2.0})
        metrics, step = t.fabric.calls[-1]
        self.assertEqual(metrics, {"train/loss": 2.0})
        self.assertEqual(step, 7)

    def test_prefix_without_pipeline_run_falls_back_to_legacy(self):
        t = _make_trainer("stage_1")  # prefix set but no _pipeline_run
        t.global_step = 4
        t._log({"train/loss": 1.0})
        metrics, step = t.fabric.calls[-1]
        self.assertEqual(metrics, {"train/loss": 1.0})
        self.assertEqual(step, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
