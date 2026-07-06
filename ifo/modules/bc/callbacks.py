from time import time
from typing import Optional, Union

import torch
from gymnasium.vector import VectorEnv
from torchmetrics import MeanMetric

from ifo.common.collector import Collector, ExplorationMode
from ifo.common.module import LightningModule
from ifo.common.trainer import SupervisedTrainer
from ifo.common.utils.utility import restore_torch_state


class RolloutCallback:
    def __init__(
        self,
        env: VectorEnv,
        rollout_steps: Union[int, float],
        rollout_frequency: Union[int, float],
        rollout_exploration_mode: ExplorationMode,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize rollout callback.

        This callback is used to evaluate the model's performance in the environment by running rollouts.
        It runs periodically during validation and collects statistics about the model's performance,
        such as episode returns and lengths.

        Args:
            env (VectorEnv): The environment to run rollouts on.
            rollout_steps (Union[int, float]): Number of steps to collect during rollouts.
            rollout_frequency (Union[int, float]): How often the callback is run every validation.
            rollout_exploration_mode (ExplorationMode): Mode to use for exploration.
            random_seed (Optional[int]): Random seed to set for the environment for each reset.
        """
        self.env = env
        self.num_envs = env.unwrapped.num_envs
        self.rollout_steps = int(rollout_steps)
        self.rollout_frequency = int(rollout_frequency)
        self.rollout_exploration_mode = rollout_exploration_mode
        self.num_rollouts = 0
        self.random_seed = random_seed

    @restore_torch_state
    @torch.no_grad()
    def _rollout(self, trainer: SupervisedTrainer, model: LightningModule, **kwargs) -> None:
        """Rollout loop running after validation has ended.

        Args:
            trainer (SupervisedTrainer): The trainer class of the model.
            model (LightningModule): The LightningModule to evaluate
        """
        model.eval()
        episode_returns, episode_lengths = MeanMetric(), MeanMetric()
        collector = Collector(
            self.env, model, random_seed=self.random_seed + trainer.fabric.global_rank * self.env.num_envs
        )
        pbar = trainer.tqdm_wrapper(
            range(self.rollout_steps),
            total=self.rollout_steps,
            desc="Rollout Callback",
            unit="steps",
            position=1,
            leave=False,
        )

        start_time = time()
        _, episode_return, episode_length = collector.rollout(
            max_steps=self.rollout_steps // (self.num_envs * trainer.fabric.world_size),
            callback=lambda x: pbar.update(self.num_envs * trainer.fabric.world_size),
            exploration_mode=self.rollout_exploration_mode,
            reset_before_rollout=True,
            store_data=False,
        )

        trainer._log(
            {"val/env_throughput": self.rollout_steps * self.num_envs * trainer.fabric.world_size / (time() - start_time)}
        )

        # Get and log training rewards and episode lengths
        if episode_return is not None:
            episode_returns(episode_return)
            episode_lengths(episode_length)
            trainer._log(
                {
                    "val/episode_return": episode_returns.compute(),
                    "val/episode_length": episode_lengths.compute(),
                }
            )
            trainer._format_iterable(pbar, {"return": episode_returns.compute()}, "val")

        self.num_rollouts += 1
        trainer.close_tqdm_wrapper(pbar, self.rollout_steps)

    def val_loop_end(self, trainer: SupervisedTrainer, model: LightningModule, **kwargs) -> None:
        """Rollout loop running after validation has ended.

        Args:
            trainer (SupervisedTrainer): The trainer class of the model.
            model (LightningModule): The LightningModule to evaluate
        """
        # No rollout if env wasn't passed.
        if self.env is None:
            return

        self._rollout(trainer=trainer, model=model, **kwargs)
