import os

import hydra
import torch
from omegaconf import DictConfig

from ifo.common.experiment import Experiment
from ifo.common.transforms.debug import get_debug_transform
from ifo.common.utils.hydra import hydra_main_multistage
from ifo.common.utils.utility import Conditional, get_latest_checkpoint, requires_grad


class ILPOExperiment(Experiment):
    """Class for ILPO experiment.

    Paper: https://arxiv.org/abs/1805.07914
    """

    stages = ["stage_1", "stage_2"]
    config = {
        "stage_1": "ilpo_default_stage_1",
        "stage_2": "ilpo_default_stage_2",
    }

    def stage_1(self, cfg: DictConfig) -> None:
        """Start training of latent policy.

        Args:
            cfg (DictConfig): Configuration for the experiment.
        """
        # Initialize Fabric
        fabric = self._configure_fabric(cfg)

        # Print config.
        self._print_config(cfg)

        # Configure logging.
        self._configure_wandb_logging(fabric, cfg)

        # Configure environment.
        env = hydra.utils.instantiate(cfg.env, num_envs=1)
        action_space = env.single_action_space
        observation_space = env.single_observation_space
        del env

        # Add callbacks.
        decoder_callback = hydra.utils.instantiate(
            cfg.decoder_callback,
            action_space=action_space,
            limit_train_batches=min(cfg.decoder_callback.limit_train_batches, cfg.trainer.limit_train_batches),
            limit_val_batches=min(cfg.decoder_callback.limit_val_batches, cfg.trainer.limit_val_batches),
        )
        fabric._callbacks.append(decoder_callback)

        # Initialize model.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space)
        module = hydra.utils.instantiate(cfg.module, net=net, debug_transform=get_debug_transform(cfg.dataset.name))

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
        """Train action decoder.

        Args:
            cfg (DictConfig): Configuration for the experiment.
        """
        # Initialize Fabric
        fabric = self._configure_fabric(cfg)

        # Print config.
        self._print_config(cfg)

        # Configure logging on rank 0.
        self._configure_wandb_logging(fabric, cfg)

        # Configure environment.
        record_video = fabric.is_global_zero  # Only record video on rank 0.
        train_env = hydra.utils.instantiate(cfg.env)
        val_env = hydra.utils.instantiate(cfg.env, record_video=record_video, run_id=cfg.stage_run_id)
        action_space = train_env.single_action_space
        observation_space = train_env.single_observation_space

        # Initialize model and load trained inverse dynamics model and latent policy.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
        checkpoint_path = get_latest_checkpoint(cfg.trainer.previous_stage_checkpoint)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)["model"]
            net.encoder.load_state_dict(self.filter_state_dict_by_prefix(state_dict, "net.encoder"))
            net.mlp.load_state_dict(self.filter_state_dict_by_prefix(state_dict, "net.mlp"))
            net.world_model.load_state_dict(self.filter_state_dict_by_prefix(state_dict, "net.world_model"))
            requires_grad(net.encoder, False)
            requires_grad(net.mlp, False)
            requires_grad(net.world_model, False)
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
    version_base=None, config_path=f"{os.getcwd()}/experiments/configs", config_name=ILPOExperiment.config
)
def main(cfg: DictConfig, stage: str):
    experiment = ILPOExperiment()
    getattr(experiment, stage)(cfg)


if __name__ == "__main__":
    main()
