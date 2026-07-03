import os

import hydra
import torch
from omegaconf import DictConfig

from ifo.common.utils.hydra import hydra_main_multistage
from ifo.common.utils.utility import (
    Conditional,
    add_legacy_features_to_vector_wrapper,
    get_latest_checkpoint,
    requires_grad,
)
from ifo.modules.lapo.experiment import LAPOExperiment


class LAPOBCExperiment(LAPOExperiment):
    """Class for LAPO experiment, where the last stage is not PPO, but behavior cloning fine-tuning.

    Paper: https://arxiv.org/abs/2312.10812
    """

    config = {
        "stage_1": "lapo_default_stage_1",
        "stage_2": "lapo_default_stage_2",
        "stage_3": "lapo_bc_default_stage_3",
    }

    def stage_3(self, cfg: DictConfig) -> None:
        """Train policy.

        Args:
            module (LightningModule, optional): Module containing the trained latent policy.
        """
        # Initialize Fabric
        fabric = self._configure_fabric(cfg)

        # Print config.
        self._print_config(cfg)

        # Configure logging
        self._configure_wandb_logging(fabric, cfg)

        # Configure environment.
        val_env = hydra.utils.instantiate(cfg.env, record_video=True, run_id=cfg.stage_run_id)
        action_space = val_env.single_action_space
        observation_space = val_env.single_observation_space

        # Add callbacks.
        rollout_callback = hydra.utils.instantiate(
            cfg.rollout_callback, env=val_env, random_seed=cfg.trainer.random_seed
        )
        fabric._callbacks.append(rollout_callback)

        # Initialize model and load trained inverse dynamics model and latent policy.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
        checkpoint_path = get_latest_checkpoint(cfg.trainer.previous_stage_checkpoint)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)["model"]
            net.latent_policy.load_state_dict(self.filter_state_dict_by_prefix(state_dict, "net.latent_policy"))
            requires_grad(net.latent_policy, False)
        module = hydra.utils.instantiate(cfg.module, net=net)

        # Initialize datasets.
        train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
        val_dataset = hydra.utils.instantiate(cfg.dataset, split="test")

        # Take subset of dataset for behavior cloning.
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(min(int(cfg.dataset.subset_size), len(train_dataset))))
        val_dataset = torch.utils.data.Subset(
            val_dataset, range(min(int(cfg.dataset.subset_size), len(val_dataset))))

        # Initialize trainer and start training.
        checkpoint_dir = f"{cfg.trainer.checkpoint_dir}/{cfg.stage_run_id}"
        trainer = hydra.utils.instantiate(cfg.trainer, fabric=fabric, checkpoint_dir=checkpoint_dir)
        trainer.fit(
            model=module, train_dataset=train_dataset, val_dataset=val_dataset, checkpoint=cfg.trainer.checkpoint_path
        )

        # Close environment.
        val_env.close()


@hydra_main_multistage(
    version_base=None, config_path=f"{os.getcwd()}/experiments/configs", config_name=LAPOBCExperiment.config
)
def main(cfg: DictConfig, stage: str):
    experiment = LAPOBCExperiment()
    add_legacy_features_to_vector_wrapper()
    getattr(experiment, stage)(cfg)


if __name__ == "__main__":
    main()
