"""
Segmentation-Based Latent Action Learning (SLAPO) experiment runner.

SLAPO extends LAPO (Latent Action Policies from Observation) by operating on masked images,
where only the agent's body pixels are visible and the background is completely masked out.
This segmentation-based approach forces the model to focus solely on the agent's pose and
movement, making it more robust to visual distractors in the environment.

Base Paper (LAPO): https://arxiv.org/abs/2312.10812

This script runs a SLAPO experiment to train a policy by learning from expert demonstrations
without requiring action labels. The policy is trained in multiple stages:
0. Training a UNet-based binary segmentation model on ground-truth masks (stage_0).
1. Training an inverse dynamics model (IDM) on predicted masked images (stage_1).
2. Training a latent policy on top of the learned latent actions (stage_2).
3. Behavior cloning fine-tuning of the policy using the learned latent actions (stage_3).
"""

import coolname

from ifo.common.runner import Runner
from ifo.modules.slapo.experiment_segmentation import SLAPOSegmentationExperiment

if __name__ == "__main__":
    run_id = coolname.generate_slug(3)
    checkpoint_dir = "./checkpoints"
    runner = Runner(SLAPOSegmentationExperiment)
    default_global_args = [
        f"run_id={run_id}",
        f"trainer.checkpoint_dir={checkpoint_dir}"
    ]
    runner.run(default_global_args=default_global_args)
