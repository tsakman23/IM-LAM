from typing import Optional, Union

import hydra
import torch
from gymnasium import spaces
from lightning import LightningModule
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from ifo.common.buffer import ReplayBuffer
from ifo.common.trainer import SupervisedTrainer
from ifo.common.utils.utility import restore_torch_state
from ifo.modules.lapo.module import LAPOPPOModule
from ifo.modules.ppo.trainer import PPOTrainer


class DecoderCallback:
    """Trains a small fully connected neural network to predict the true action given the latent action
    for supervised training.
    """

    def __init__(
        self,
        action_space: spaces.Space,
        net: DictConfig,
        module: DictConfig,
        max_epochs: int = 1,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
    ):
        """Instantiate the decoder callback.

        Args:
            action_space (spaces.Space): The action space of the environment.
            net (DictConfig): The configuration for the decoder network.
            module (DictConfig): The configuration for the decoder module.
            max_epochs (int): The maximum number of epochs to train the decoder.
            limit_train_batches (Union[int, float], optional): The maximum number of batches to train the decoder.
            limit_val_batches (Union[int, float], optional): The maximum number of batches to validate the decoder.
        """
        self.action_space = action_space
        self.net = net
        self.module = module
        self.max_epochs = max_epochs
        self.limit_train_batches = int(limit_train_batches) if limit_train_batches is not None else float("inf")
        self.limit_val_batches = int(limit_val_batches) if limit_val_batches is not None else float("inf")

    def _train_loop(
        self,
        trainer: SupervisedTrainer,
        model: LightningModule,
        decoder: LightningModule,
        optimizer: Optimizer,
        train_loader: DataLoader,
    ) -> None:
        """The decoder loop running to train the decoder.

        Args:
            trainer (SupervisedTrainer): The trainer of the model.
            model (LightningModule): The inverse dynamics model.
            decoder (LightningModule): The decoder to train.
            optimizer (Optimizer): The optimizer to use for training the decoder.
            train_loader (DataLoader): The dataloader yielding the training batches.
        """
        # Only train action decoder.
        model.eval()
        decoder.train()

        # Train decoder for max_epochs epochs.
        train_loss = MeanMetric().to(trainer.fabric.device)
        steps = min(len(train_loader), self.limit_train_batches)
        pbar = trainer.tqdm_wrapper(
            train_loader, total=steps * self.max_epochs, desc="Decoder Callback: Training Loop", position=1, leave=False
        )
        for epoch in range(self.max_epochs):
            for batch_idx, batch in enumerate(pbar):
                # End epoch if max batches for this epoch is reached.
                if batch_idx >= steps:
                    break

                with torch.no_grad():
                    prediction = model.predict_step(batch, batch_idx, steps)
                    prediction["action"] = batch["action"]
                step_dict = decoder.training_step(prediction, batch_idx, steps)
                optimizer.zero_grad()
                trainer.fabric.backward(step_dict["loss"])
                optimizer.step()
                train_loss(step_dict["loss"])
                trainer._format_iterable(pbar, step_dict["loss"], "train")
        trainer.fabric.log_dict({"train/action_decoder_loss": train_loss.compute()}, step=trainer.global_step)
        trainer.close_tqdm_wrapper(pbar, steps * self.max_epochs)

    def _val_loop(
        self,
        trainer: SupervisedTrainer,
        model: LightningModule,
        decoder: LightningModule,
        val_loader: DataLoader,
    ) -> None:
        """The decoder loop running to validate the decoder.

        Args:
            trainer (SupervisedTrainer): The trainer of the model.
            model (LightningModule): The inverse dynamics model.
            decoder (LightningModule): The decoder to validate.
            val_loader (DataLoader): The dataloader yielding the validation batches.
        """
        # Validate the decoder.
        model.eval()
        decoder.eval()

        mean_metrics = {}
        steps = min(len(val_loader), self.limit_val_batches)
        pbar = trainer.tqdm_wrapper(
            val_loader, total=steps, desc="Decoder Callback: Validation Loop", position=1, leave=False
        )
        for batch_idx, batch in enumerate(pbar):
            # End epoch if max batches for this epoch is reached.
            if batch_idx >= steps:
                break

            with torch.no_grad():
                prediction = model.predict_step(batch, batch_idx, steps)
                prediction["action"] = batch["action"]
                step_dict = decoder.validation_step(prediction, batch_idx, steps)

            if not mean_metrics:
                mean_metrics = {key: MeanMetric().to(trainer.fabric.device) for key in step_dict.keys()}
            for k, metric in mean_metrics.items():
                metric(step_dict[k])

            trainer._format_iterable(pbar, step_dict["loss"], "val")

        trainer.fabric.log_dict(
            {f"val/action_decoder_{k}": v.compute() for k, v in mean_metrics.items()}, step=trainer.global_step
        )
        trainer.close_tqdm_wrapper(pbar, steps)

    @restore_torch_state
    def val_loop_end(
        self,
        trainer: SupervisedTrainer,
        model: LightningModule,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        **kwargs,
    ) -> None:
        """The decoder loop running to check if the learned codes have information about the true actions.

        Args:
            trainer (SupervisedTrainer): The trainer of the model.
            model (LightningModule): The inverse dynamics model.
            train_loader (DataLoader): The dataloader yielding the training batches.
            val_loader (DataLoader, optional): The dataloader yielding the validation batches.

        Raises:
            ValueError: If either train_loader or val_loader are not provided.
        """
        net = hydra.utils.instantiate(self.net, _partial_=True)(action_space=self.action_space)
        decoder = hydra.utils.instantiate(self.module, net=net, _recursive_=False)
        optimizer, _ = decoder.configure_optimizers()
        decoder, optimizer = trainer.fabric.setup(decoder, optimizer)

        self._train_loop(trainer, model, decoder, optimizer, train_loader)
        if val_loader:
            self._val_loop(trainer, model, decoder, val_loader)


class DecoderCallbackPPO:
    """Trains a small fully connected neural network to predict the true action given the latent action
    for reinforcement learning with PPO.
    """

    def __init__(
        self,
        optimizer: DictConfig,
        max_epochs: int = 1,
        batch_size: int = 512,
        num_workers: int = 2,
        dataset_size: Union[int, float] = 10000,
        start_steps: Union[int, float] = 10000,
        end_steps: Union[int, float] = 400000,
        end_update_every_iteration_steps: Union[int, float] = 50000,
        update_frequency: Union[int, float] = 20,
        max_updates: Optional[Union[int, float]] = 100,
        device: str = "cpu",
        num_envs: int = 1,
        idm_block_size: int = 2,
    ):
        """Instantiate the decoder callback.

        Args:
            optimizer (DictConfig): The optimizer to use for the decoder.
            max_epochs (int): The maximum number of epochs to train the decoder.
            batch_size (int): The batch size of the dataloader for the decoder.
            num_workers (int): The number of workers for the dataloader.
            dataset_size (Union[int, float], optional): The maximum number of samples to store in the dataset. If
                the dataset exceeds this size, the oldest samples are removed.
            start_steps (Union[int, float]): The number of steps in the environment before starting to train the
                decoder.
            end_steps (Union[int, float]): The maximum number of steps in the environment till the the decoder is
                trained.
            end_update_every_iteration_steps (Union[int, float]): The number of steps in the environment to update the
                decoder every iteration.
            update_frequency (Union[int, float]): The frequency of iterations to update the decoder.
            max_updates (Union[int, float], optional): The maximum number of updates to train the decoder.
            device (str): The device to save the dataset to.
            num_envs (int): The number of environments to collect data from.
            idm_block_size (int): The block size of the IDM.
        """
        self.optimizer_cfg = optimizer
        self.optimizer = None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_size = int(dataset_size)
        self.start_steps = int(start_steps)
        self.end_steps = int(end_steps)
        self.end_update_every_iteration_steps = int(end_update_every_iteration_steps)
        self.update_frequency = int(update_frequency)
        self.max_updates = int(max_updates) if max_updates else max_updates
        self.device = device
        self.data_buffer = ReplayBuffer(
            capacity=self.dataset_size,
            num_envs=num_envs,
            block_size=idm_block_size,
            device=self.device,
            is_frame_stacked=True,
        )

    @torch.no_grad()
    def rollout_loop_end(self, rollout_data: TensorDict, **kwargs) -> None:
        """Callback that stores rollout data in the replay buffer for training the decoder.

        This method is called at the end of each rollout loop to collect state-action pairs
        that will be used to train the decoder to predict actions from latent states.

        Args:
            rollout_data (TensorDict): Dictionary containing the rollout data including
                observations, actions, and other environment information.
            **kwargs: Additional keyword arguments passed by the trainer.
        """
        self.data_buffer.extend(rollout_data)

    @restore_torch_state
    def train_loop_end(self, trainer: PPOTrainer, model: LAPOPPOModule, **kwargs) -> None:
        """The decoder loop training the decoder supervised to decode latent into true actions.

        Args:
            trainer (PPOTrainer): The trainer of the model.
            model (LAPOPPOModule): The LightningModule to evaluate
        """
        # Train decoder only if needed.
        is_interval = self.start_steps < trainer.global_step < self.end_steps
        is_update_every_iteration = trainer.global_step < self.end_update_every_iteration_steps
        is_regular_update = trainer.global_iteration % self.update_frequency == 0
        if not (is_interval and (is_update_every_iteration or is_regular_update)):
            return

        # Instantiate optimizer if not done yet.
        if self.optimizer is None:
            self.optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=model.net.decoder.parameters())
            self.optimizer = trainer.fabric.setup_optimizers(self.optimizer)

        dataset = self.data_buffer.dataset
        dataloader = model.training_dataloader(dataset)
        dataloader = trainer.fabric.setup_dataloaders(dataloader)

        # Train only decoder.
        model.eval()
        model.net.decoder.train()

        # Train decoder for max_epochs epochs.
        pbar = trainer.tqdm_wrapper(
            dataloader,
            total=len(dataloader) * self.max_epochs,
            desc="Decoder Callback: Training Loop",
            position=1,
            leave=False,
        )
        train_loss = MeanMetric().to(trainer.fabric.device)
        for epoch in range(self.max_epochs):
            for batch_idx, batch in enumerate(pbar):
                step_dict = model.predict_step(batch, batch_idx, len(dataloader))
                self.optimizer.zero_grad()
                trainer.fabric.backward(step_dict["loss"])
                self.optimizer.step()
                train_loss(step_dict["loss"])
                trainer._format_iterable(pbar, step_dict["loss"], "train")

        trainer.fabric.log_dict({"train/action_decoder_loss": train_loss.compute()}, step=trainer.global_step)
