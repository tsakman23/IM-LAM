"""
DreamerV3 experiment runner.

Paper: https://arxiv.org/abs/2301.04104

This script runs a DreamerV3 experiment. DreamerV3 is a general reinforcement
learning algorithm that learns a world model of the environment and improves its
behavior by imagining future scenarios. It is designed to be robust and applicable
to a wide range of tasks with a single configuration. Notably, it was the first
algorithm to solve the Minecraft diamond challenge from scratch.
"""

import coolname

from ifo.common.runner import Runner
from ifo.modules.dreamerv3.experiment import DreamerV3Experiment

if __name__ == "__main__":
    run_id = coolname.generate_slug(3)
    runner = Runner(DreamerV3Experiment)
    default_global_args = [f"run_id={run_id}"]
    runner.run(default_global_args=default_global_args)
