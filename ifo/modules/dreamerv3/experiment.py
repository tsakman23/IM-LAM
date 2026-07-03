import os

import hydra
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from omegaconf import DictConfig

from ifo.common.env.wrapper import DictObservationWrapper, OneHotActionWrapper, UUIDWrapper
from ifo.common.experiment import Experiment
from ifo.common.transforms.debug import get_debug_transform
from ifo.common.utils.hydra import hydra_main_multistage
from ifo.common.utils.utility import Conditional, add_legacy_features_to_vector_wrapper


class DreamerV3Experiment(Experiment):
    """Class for DreamerV3 experiment.

    Paper: https://arxiv.org/abs/2301.04104
    """

    stages = ["stage_1"]
    config = {
        "stage_1": "dreamerv3_default",
    }

    def stage_1(self, cfg: DictConfig) -> None:
        """Train DreamerV3 policy."""
        # Initialize Fabric
        fabric = self._configure_fabric(cfg)

        # Print config.
        self._print_config(cfg)

        # Configure logging.
        self._configure_wandb_logging(fabric, cfg)

        # Configure environment.
        record_video = fabric.is_global_zero  # Only record video on rank 0.
        train_env = self.dreamer_env_wrapper(hydra.utils.instantiate(cfg.env))
        val_env = self.dreamer_env_wrapper(
            hydra.utils.instantiate(cfg.env, record_video=record_video, run_id=cfg.stage_run_id)
        )
        action_space = train_env.single_action_space
        observation_space = train_env.single_observation_space

        # Initialize model.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
        module = hydra.utils.instantiate(cfg.module, net=net, debug_transform=get_debug_transform(cfg.env.name))

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

    def dreamer_env_wrapper(self, env: VectorEnv) -> VectorEnv:
        """Add necessarywrappers for DreamerV3 to the environment."""
        env = DictObservationWrapper(env, add_is_first=True, add_is_terminal=True, add_continue=True)
        if isinstance(env.single_action_space, spaces.Discrete):
            env = OneHotActionWrapper(env)
        env = UUIDWrapper(env)
        return env


@hydra_main_multistage(
    version_base=None, config_path=f"{os.getcwd()}/experiments/configs", config_name=DreamerV3Experiment.config
)
def main(cfg: DictConfig, stage: str):
    experiment = DreamerV3Experiment()
    add_legacy_features_to_vector_wrapper()
    getattr(experiment, stage)(cfg)


if __name__ == "__main__":
    main()
