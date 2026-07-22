"""Tests for the per-task Var(a) lookup used to compute action_decoder_nmse.

Run:  conda_env/bin/python tests/test_expert_constants.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ifo.common.utils.expert_constants import (  # noqa: E402
    ACTION_VARIANCE,
    get_action_variance,
    task_slug,
)


class TaskSlugTest(unittest.TestCase):
    def test_strips_meta_world_mt1_and_masked_prefixes(self):
        self.assertEqual(task_slug("Meta-World/masked-MT1-push-v3"), "push-v3")

    def test_strips_distracting_prefix(self):
        self.assertEqual(task_slug("Meta-World/distracting-MT1-push-v3"), "push-v3")

    def test_bare_slug_is_unchanged(self):
        self.assertEqual(task_slug("push-v3"), "push-v3")


class GetActionVarianceTest(unittest.TestCase):
    def test_known_maskLAM_task(self):
        self.assertAlmostEqual(
            get_action_variance("Meta-World/masked-MT1-push-v3"), ACTION_VARIANCE["push-v3"]
        )

    def test_known_regenerated_task(self):
        self.assertAlmostEqual(
            get_action_variance("Meta-World/masked-MT1-sweep-into-v3"),
            ACTION_VARIANCE["sweep-into-v3"],
        )

    def test_dropped_task_returns_none(self):
        # dial-turn-v3 is intentionally absent: the scripted expert never succeeds,
        # so there is no usable NMSE denominator for it.
        self.assertIsNone(get_action_variance("Meta-World/masked-MT1-dial-turn-v3"))

    def test_unknown_task_returns_none_not_keyerror(self):
        self.assertIsNone(get_action_variance("Meta-World/masked-MT1-not-a-real-task-v3"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
