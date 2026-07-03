"""
Latent Action Policies from Observation (LAPO) experiment runner.

Paper: https://arxiv.org/abs/2312.10812

This script runs a LAPO experiment to train a policy by learning from expert demonstrations
without requiring action labels. The policy is trained in multiple stages:
1. Initial training of an inverse dynamics model from a video dataset.
2. Training of a latent policy from the inverse dynamics model.
3. Fine-tuning the policy through interaction with the environment.
"""

import coolname

from ifo.common.runner import Runner
from ifo.modules.lapo.experiment import LAPOExperiment

if __name__ == "__main__":
    run_id = coolname.generate_slug(3)
    checkpoint_dir = "./checkpoints"
    runner = Runner(LAPOExperiment)
    default_global_args = [
        f"run_id={run_id}",
        f"trainer.checkpoint_dir={checkpoint_dir}",
    ]
    runner.run(default_global_args=default_global_args)
