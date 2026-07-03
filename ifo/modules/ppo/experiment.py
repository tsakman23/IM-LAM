import os

import hydra
from omegaconf import DictConfig

from ifo.common.experiment import Experiment
from ifo.common.utils.hydra import hydra_main_multistage
from ifo.common.utils.utility import Conditional, add_legacy_features_to_vector_wrapper


class PPOExperiment(Experiment):
    """Class for proximal policy optimization (PPO) experiment.

    Paper: https://arxiv.org/abs/1707.06347
    """

    stages = ["stage_1"]
    config = {
        "stage_1": "ppo_default",
    }

    def stage_1(self, cfg: DictConfig) -> None:
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

        # Initialize model.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
        module = hydra.utils.instantiate(cfg.module, net=net)

        # Initialize trainer and start training.
        checkpoint_dir = f"{cfg.trainer.checkpoint_dir}/{cfg.stage_run_id}"
        trainer = hydra.utils.instantiate(
            cfg.trainer, fabric=fabric, checkpoint_dir=checkpoint_dir, random_seed=cfg.trainer.random_seed
        )
        trainer.fit(
            model=module,
            train_env=train_env,
            val_env=val_env,
            checkpoint=cfg.trainer.checkpoint_path,
        )

        # Close environment.
        train_env.close()
        val_env.close()


@hydra_main_multistage(
    version_base=None, config_path=f"{os.getcwd()}/experiments/configs", config_name=PPOExperiment.config
)
def main(cfg: DictConfig, stage: str):
    experiment = PPOExperiment()
    add_legacy_features_to_vector_wrapper()
    getattr(experiment, stage)(cfg)


if __name__ == "__main__":
    main()
