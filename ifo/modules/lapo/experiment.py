import os

import hydra
import torch
from omegaconf import DictConfig

from ifo.common.experiment import Experiment
from ifo.common.transforms.debug import get_debug_transform
from ifo.common.utils.expert_constants import get_action_variance
from ifo.common.utils.hydra import hydra_main_multistage
from ifo.common.utils.utility import (
    Conditional,
    add_legacy_features_to_vector_wrapper,
    get_latest_checkpoint,
    requires_grad,
)


class LAPOExperiment(Experiment):
    """Class for LAPO experiment.

    Paper: https://arxiv.org/abs/2312.10812
    """

    stages = ["stage_1", "stage_2", "stage_3"]
    config = {"stage_1": "lapo_default_stage_1", "stage_2": "lapo_default_stage_2", "stage_3": "lapo_default_stage_3"}

    def stage_1(self, cfg: DictConfig) -> None:
        """Start training of inverse dynamics model.

        Args:
            cfg (DictConfig): Configuration for the experiment.
        """
        # Initialize Fabric
        fabric = self._configure_fabric(cfg)

        # Print config.
        self._print_config(cfg)

        # Configure logging
        self._configure_wandb_logging(fabric, cfg)

        # Configure environment.
        env = hydra.utils.instantiate(cfg.env, num_envs=1)
        action_space = env.single_action_space
        observation_space = env.single_observation_space
        del env

        # Initialize model.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
        # Var(a) for action_decoder_nmse: auto-resolved from the task in cfg.env.name via the
        # ACTION_VARIANCE table (ifo.common.utils.expert_constants), unless a config/CLI override
        # (module.action_variance=...) is already set, which takes precedence. Mirrors SLAPOExperiment.
        action_variance = cfg.module.get("action_variance")
        if action_variance is None:
            action_variance = get_action_variance(cfg.env.name)
        module = hydra.utils.instantiate(
            cfg.module, net=net, debug_transform=get_debug_transform(cfg.dataset.name),
            action_variance=action_variance,
        )

        # Initialize datasets.
        train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
        val_dataset = hydra.utils.instantiate(cfg.dataset, split="test")

        # Initialize trainer and start training.
        checkpoint_dir = f"{cfg.trainer.checkpoint_dir}/{cfg.stage_run_id}"
        trainer = hydra.utils.instantiate(cfg.trainer, fabric=fabric, checkpoint_dir=checkpoint_dir)
        trainer.fit(
            model=module, train_dataset=train_dataset, val_dataset=val_dataset, checkpoint=cfg.trainer.checkpoint_path
        )

    def stage_2(self, cfg: DictConfig) -> None:
        """Train latent policy.

        Args:
            cfg (DictConfig): Configuration for the experiment.
        """
        # Initialize Fabric
        fabric = self._configure_fabric(cfg)

        # Print config.
        self._print_config(cfg)

        # Configure logging
        self._configure_wandb_logging(fabric, cfg)

        # Configure environment.
        env = hydra.utils.instantiate(cfg.env, num_envs=1)
        action_space = env.single_action_space
        observation_space = env.single_observation_space
        del env

        # Initialize model and load trained inverse dynamics model.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
        checkpoint_path = get_latest_checkpoint(cfg.trainer.previous_stage_checkpoint)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)["model"]
            net.inverse_dynamics_model.load_state_dict(self.filter_state_dict_by_prefix(state_dict, "net.encoder"))
            requires_grad(net.inverse_dynamics_model, False)
        module = hydra.utils.instantiate(cfg.module, net=net)

        # Initialize datasets.
        train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
        val_dataset = hydra.utils.instantiate(cfg.dataset, split="test")

        # Initialize trainer and start training.
        checkpoint_dir = f"{cfg.trainer.checkpoint_dir}/{cfg.stage_run_id}"
        trainer = hydra.utils.instantiate(cfg.trainer, fabric=fabric, checkpoint_dir=checkpoint_dir)
        trainer.fit(
            model=module, train_dataset=train_dataset, val_dataset=val_dataset, checkpoint=cfg.trainer.checkpoint_path
        )

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
        train_env = hydra.utils.instantiate(cfg.env)
        val_env = hydra.utils.instantiate(cfg.env, record_video=True, run_id=cfg.stage_run_id)
        action_space = train_env.single_action_space
        observation_space = train_env.single_observation_space

        # Add callbacks.
        decoder_callback = hydra.utils.instantiate(
            cfg.decoder_callback, num_envs=cfg.env.num_envs, idm_block_size=cfg.net.idm_block_size
        )
        fabric._callbacks.append(decoder_callback)

        # Initialize model and load trained inverse dynamics model and latent policy.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
        checkpoint_path = get_latest_checkpoint(cfg.trainer.previous_stage_checkpoint)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)["model"]
            net.inverse_dynamics_model.load_state_dict(
                self.filter_state_dict_by_prefix(state_dict, "net.inverse_dynamics_model")
            )
            net.latent_policy.load_state_dict(self.filter_state_dict_by_prefix(state_dict, "net.latent_policy"))
            requires_grad(net.inverse_dynamics_model, False)
            requires_grad(net.latent_policy, False)
            net.clone_fc()
        module = hydra.utils.instantiate(cfg.module, net=net)

        # Initialize trainer and start training.
        checkpoint_dir = f"{cfg.trainer.checkpoint_dir}/{cfg.stage_run_id}"
        trainer = hydra.utils.instantiate(
            cfg.trainer, fabric=fabric, checkpoint_dir=checkpoint_dir, random_seed=cfg.trainer.random_seed
        )
        trainer.fit(model=module, train_env=train_env, val_env=val_env, checkpoint=cfg.trainer.checkpoint_path)

        # Close environment.
        train_env.close()
        val_env.close()


@hydra_main_multistage(
    version_base=None, config_path=f"{os.getcwd()}/experiments/configs", config_name=LAPOExperiment.config
)
def main(cfg: DictConfig, stage: str):
    experiment = LAPOExperiment()
    add_legacy_features_to_vector_wrapper()
    getattr(experiment, stage)(cfg)


if __name__ == "__main__":
    main()
