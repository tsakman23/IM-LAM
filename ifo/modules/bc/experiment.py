import os

import hydra
from omegaconf import DictConfig

from ifo.common.experiment import Experiment
from ifo.common.utils.hydra import hydra_main_multistage
from ifo.common.utils.utility import Conditional


class BCExperiment(Experiment):
    """Class for behavior cloning experiment."""

    stages = ["stage_1"]
    config = {
        "stage_1": "bc_default",
    }

    def stage_1(self, cfg: DictConfig) -> None:
        """Start training of behavior cloning model.

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
        record_video = fabric.is_global_zero  # Only record video on rank 0.
        val_env = hydra.utils.instantiate(cfg.env, record_video=record_video, run_id=cfg.stage_run_id)
        action_space = val_env.single_action_space
        observation_space = val_env.single_observation_space

        # Add callbacks.
        rollout_callback = hydra.utils.instantiate(
            cfg.rollout_callback, env=val_env, random_seed=cfg.trainer.random_seed
        )
        fabric._callbacks.append(rollout_callback)

        # Initialize model.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
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

        # Close environment.
        val_env.close()


@hydra_main_multistage(
    version_base=None, config_path=f"{os.getcwd()}/experiments/configs", config_name=BCExperiment.config
)
def main(cfg: DictConfig, stage: str):
    experiment = BCExperiment()
    getattr(experiment, stage)(cfg)


if __name__ == "__main__":
    main()
