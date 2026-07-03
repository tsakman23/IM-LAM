"""
Proximal Policy Optimization (PPO) experiment runner.

Paper: https://arxiv.org/abs/1707.06347

This script runs a PPO experiment to train a policy through direct interaction with the environment.
PPO is a policy gradient method that uses a clipped objective function to prevent too large policy updates,
making it more stable and sample-efficient than traditional policy gradient methods.
"""

import coolname

from ifo.common.runner import Runner
from ifo.modules.ppo.experiment import PPOExperiment

if __name__ == "__main__":
    run_id = coolname.generate_slug(3)
    runner = Runner(PPOExperiment)
    default_global_args = [f"run_id={run_id}"]
    runner.run(default_global_args=default_global_args)
