"""
Imitating Latent Policies from Observation (ILPO) experiment runner.

Paper: https://arxiv.org/abs/1805.07914

This script runs an ILPO experiment to train a policy by learning from expert demonstrations
and improving it through policy optimization. The policy is trained in multiple stages:
1. Initial training of a latent policy from a video dataset.
2. Further refinement by interacting with the environment to learn the mapping between latent actions and real actions.
"""

import coolname

from ifo.common.runner import Runner
from ifo.modules.ilpo.experiment import ILPOExperiment

if __name__ == "__main__":
    run_id = coolname.generate_slug(3)
    checkpoint_dir = "./checkpoints"
    runner = Runner(ILPOExperiment)
    default_global_args = [
        f"run_id={run_id}",
        f"trainer.checkpoint_dir={checkpoint_dir}",
    ]
    runner.run(default_global_args=default_global_args)
