import glob
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from time import time
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, Union

import torch
from lightning.fabric import Fabric
from lightning_utilities import apply_to_collection
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanMetric
from tqdm import tqdm

from ifo.common.module import SupervisedLightningModule
from ifo.common.utils.utility import get_latest_checkpoint, rename_key


class LightningTrainer(ABC):
    """Lightning based Trainer class."""

    def __init__(
        self,
        fabric: Fabric,
        compile: bool = False,
        checkpoint_dir: str = "./checkpoints",
        validation_criteria: Literal["min", "max"] = "min",
        validation_unit: Literal["iteration", "epoch", "step"] = "epoch",
        random_seed: Optional[int] = None,
        log_prefix: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the supervised learning trainer.

        Args:
            fabric (Fabric): The fabric instance to use for training.
            compile (bool): Whether to compile the model.
            checkpoint_dir (str): Directory to store checkpoints to and to automatically load the latest checkpoint
                from, when available. Can be overwritten by the `checkpoint_path` argument in :meth:`fit`.
            validation_criteria (Literal["min", "max"]): The criteria to use for validation.
            validation_unit (Literal["iteration", "epoch", "step"]): The unit to use for validation.
            random_seed (Optional[int]): The random seed to use for training.
        """
        self.fabric = fabric
        self.global_step = 0
        self.global_epoch = 0
        self.global_iteration = 0
        self.should_stop = False
        self.checkpoint_dir = checkpoint_dir
        self.compile = compile
        self.validation_criteria = validation_criteria
        self.validation_unit = validation_unit
        self.random_seed = random_seed
        self.log_prefix = log_prefix
        self._current_val_loss = float("inf") if validation_criteria == "min" else -float("inf")
        self._prev_val_loss = float("inf") if validation_criteria == "min" else -float("inf")
        self._current_train_loss = float("inf") if validation_criteria == "min" else -float("inf")
        self._prev_train_loss = float("inf") if validation_criteria == "min" else -float("inf")

    def _log(self, metrics: Dict[str, Any]) -> None:
        """Log a metrics dict to the active logger.

        When ``log_prefix`` is set (in-process multi-stage pipeline), keys are prefixed
        as ``{log_prefix}/{key}`` and a per-stage, 0-based x-axis is emitted as
        ``{log_prefix}/step``; the native W&B step is intentionally left unset so a single
        run can hold several stages without a monotonic-step conflict (the orchestrator
        registers ``wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")``).
        When ``log_prefix`` is ``None``, the legacy behavior (native ``step=global_step``,
        unprefixed keys) is preserved so other experiments are unaffected.

        Args:
            metrics (Dict[str, Any]): Metric name -> value (names already carry any
                ``train/`` / ``val/`` / ``trainer/`` sub-prefix).
        """
        run = getattr(self.fabric, "_pipeline_run", None)
        if self.log_prefix and run is not None:
            # In-process pipeline: log to the raw W&B run so our per-stage
            # define_metric(f"{prefix}/*", step_metric=f"{prefix}/step") governs the x-axis
            # and wandb's internal step auto-increments monotonically across stages. (Routing
            # through the fabric WandbLogger would force every metric onto its own
            # `trainer/global_step` x-axis, which we never advance.)
            payload = {f"{self.log_prefix}/{key}": value for key, value in metrics.items()}
            payload[f"{self.log_prefix}/step"] = self.global_step
            run.log(payload)
        else:
            self.fabric.log_dict(metrics, step=self.global_step)

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """Fit the model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def train_loop(self, *args, **kwargs) -> None:
        """The training loop.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def val_loop(self, *args, **kwargs) -> None:
        """The validation loop.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        pass

    def tqdm_wrapper(self, iterable: Iterable, total: Optional[int] = None, **kwargs: Any) -> Union[tqdm, Iterable]:
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable (Iterable): The iterable to wrap with tqdm.
            total (Optional[int]): The total length of the iterable, necessary in case the number of batches
                was limited.
            **kwargs (Any): Additional keyword arguments to pass to tqdm.

        Returns:
            Union[tqdm, Iterable]: The wrapped iterable with tqdm if global rank is zero,
                otherwise the original iterable.
        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def close_tqdm_wrapper(self, iterable: Union[tqdm, Iterable], total: int) -> None:
        """Closes the tqdm wrapper and updates the remaining steps if necessary.

        Args:
            iterable (Iterable): The iterable to close the tqdm wrapper for.
        """
        if isinstance(iterable, tqdm):
            # If pbar is not completely full, fill it with the remaining steps. Then close it.
            if iterable.n < total:
                iterable.update(total - iterable.n)
            iterable.close()

    def _format_iterable(
        self,
        prog_bar: Union[tqdm, Iterable],
        candidates: Optional[Union[Tensor, Mapping[str, Union[Tensor, float, int]]]],
        prefix: Optional[str] = None,
    ) -> None:
        """Adds values as postfix string to progress bar.

        Args:
            prog_bar (Union[tqdm, Iterable]): A progress bar (on global rank zero) or an iterable (every other rank).
            candidates (Optional[Union[Tensor, Mapping[str, Union[Tensor, float, int]]]]): The values to add as postfix
                strings to the progress bar.
            prefix (str): The prefix to add to each of these values.
        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            prefix_str = (prefix + "_") if prefix else ""
            float_candidates = apply_to_collection(candidates, Tensor, lambda x: x.item())
            if isinstance(candidates, Tensor):
                postfix_str += f" {prefix_str}loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix_str}{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)

    @staticmethod
    def load_checkpoint(
        fabric: Fabric,
        state: Dict[str, Union[nn.Module, Optimizer, Any]] = {},
        default_checkpoint_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        load_only_model: bool = False,
    ) -> Tuple[int, int, int]:
        """Loads a checkpoint from the `checkpoint_dir` or a specific checkpoint path.

        Args:
            fabric (Fabric): The fabric instance to use for loading the checkpoint.
            state (Dict[str, Union[nn.Module, Optimizer, Any]]): A dictionary containing model, optimizer,
                and lr scheduler.
            default_checkpoint_dir (Optional[str]): The default checkpoint directory to load the latest checkpoint from.
            checkpoint_path (Optional[str]): The path to load the checkpoint from. If not specified, the latest
                checkpoint from `checkpoint_dir` is loaded.
            load_only_model (bool): If True, only the model is loaded from the checkpoint.
        Returns:
            Tuple[int, int, int]: The global step, epoch, and iteration.
        """
        global_step = global_epoch = global_iteration = 0

        # Try to load from a specific checkpoint path.
        checkpoint_path = get_latest_checkpoint(checkpoint_path, extension="ckpt")

        # If no checkpoint path is provided, try to continue training from a previous checkpoint.
        if checkpoint_path is None:
            checkpoint_path = get_latest_checkpoint(default_checkpoint_dir, extension="ckpt")

        # Pop everything except the model if load_only_model is True.
        if load_only_model:
            remove_keys = [k for k in state.keys() if k != "model"]
            for key in remove_keys:
                state.pop(key)

        if checkpoint_path:
            remainder = fabric.load(checkpoint_path, state)
            remainder = {} if load_only_model else remainder
            global_step = remainder.pop("global_step", 0)
            global_epoch = remainder.pop("global_epoch", 0)
            global_iteration = remainder.pop("global_iteration", 0)
            if remainder:
                print(f"Warning: Unused Checkpoint Values: {remainder.keys()}")
            print(f"Successfully loaded checkpoint {checkpoint_path}")
        else:
            print("No checkpoint path provided, starting from scratch.")
        return global_step, global_epoch, global_iteration

    def save_best(
        self,
        state: Optional[Dict[str, Union[nn.Module, Optimizer, Any]]],
        on_validation: bool = False,
        keep_last: bool = False,
    ) -> None:
        """Saves the best checkpoint based on validation or training performance.

        If a validation dataset is used, the checkpoint is saved based on validation performance.
        Otherwise, it is saved based on training performance.

        Args:
            state (Optional[Mapping]): A mapping containing model, optimizer, and lr scheduler.
            on_validation (bool): Whether to save based on validation performance.
            keep_last (bool): If True, deletes previous checkpoints and only keeps the latest one. Defaults to False.
        """
        # Calculate if the current validation result is better than the previous one.
        is_smaller_val = self._current_val_loss < self._prev_val_loss
        is_greater_val = not is_smaller_val
        is_better_val = is_smaller_val if self.validation_criteria == "min" else is_greater_val

        # Calculate if the current training result is better than the previous one.
        is_smaller_train = self._current_train_loss < self._prev_train_loss
        is_greater_train = not is_smaller_train
        is_better_train = is_smaller_train if self.validation_criteria == "min" else is_greater_train

        # Save on better validation results.
        if on_validation and is_better_val:
            self.save(state, self.validation_unit, keep_last)
        # Save on better training results if no validation is used.
        elif not on_validation and is_better_train:
            self.save(state, self.validation_unit, keep_last)

    def save(
        self,
        state: Dict[str, Union[nn.Module, Optimizer, Any]] = {},
        post_fix: Literal["iteration", "epoch", "step"] = "epoch",
        keep_last: bool = False,
    ) -> None:
        """Saves a checkpoint to the `checkpoint_dir`.

        Args:
            state (Dict[str, Union[nn.Module, Optimizer, Any]]): A dictionary containing model, optimizer,
                and lr scheduler.
            post_fix (Literal["iteration", "epoch", "step"]): The postfix to add to the checkpoint file name.
            keep_last (bool): If True, deletes previous checkpoints and only keeps the latest one. Defaults to False.
        """
        state.update(
            global_step=self.global_step, global_epoch=self.global_epoch, global_iteration=self.global_iteration
        )

        # Delete previous checkpoints if keep_last is True
        if keep_last:
            pattern = os.path.join(self.checkpoint_dir, f"{post_fix}-*.ckpt")
            existing_checkpoints = glob.glob(pattern)
            for checkpoint in existing_checkpoints:
                try:
                    os.remove(checkpoint)
                except OSError:
                    # Ignore errors if file doesn't exist or can't be deleted
                    pass
        if post_fix == "iteration":
            path = os.path.join(self.checkpoint_dir, f"iteration-{self.global_iteration:04d}.ckpt")
        elif post_fix == "epoch":
            path = os.path.join(self.checkpoint_dir, f"epoch-{self.global_epoch:04d}.ckpt")
        elif post_fix == "step":
            path = os.path.join(self.checkpoint_dir, f"step-{self.global_step:09d}.ckpt")
        else:
            raise ValueError(f"Invalid post_fix: {post_fix}, must be one of 'iteration', 'epoch', or 'step'.")
        self.fabric.save(path, state)
        self.fabric.call("save_model", **rename_key(locals(), "self", "trainer"))


class SupervisedTrainer(LightningTrainer):
    """Trainer class for supervised training."""

    def __init__(
        self,
        fabric: Fabric,
        compile: bool = False,
        checkpoint_dir: str = "./checkpoints",
        validation_criteria: Literal["min", "max"] = "min",
        validation_frequency: Union[int, float] = 1,
        validation_unit: Literal["epoch", "step"] = "epoch",
        max_epochs: Optional[Union[int, float]] = 1,
        max_steps: Optional[Union[int, float]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        **kwargs,
    ) -> None:
        """Initialize the supervised learning trainer.

        Args:
            fabric (Fabric): The fabric instance to use for training.
            compile (bool): Whether to compile the model using torch.compile.
            checkpoint_dir (str): Directory to store checkpoints to and to automatically load the latest checkpoint
                from, when available. Can be overwritten by the `checkpoint_path` argument in :meth:`fit`.
            validation_criteria (Literal["min", "max"]): The criteria to use for validation, either minimizing or
                maximizing the validation metric.
            validation_frequency (Union[int, float]): How many epochs or steps to run before each validation epoch.
            validation_unit (Literal["epoch", "step"]): Whether to validate on epochs or steps. Must be ``step`` when
                ``max_steps`` is set.
            max_epochs (Optional[Union[int, float]]): Maximum epochs when not using ``max_steps``. Defaults to ``1``.
                Must be ``None`` when ``max_steps`` is set.
            max_steps (Optional[Union[int, float]]): Total optimizer steps to train (``global_step`` target). When set,
                ``max_epochs`` must be ``None`` and ``validation_unit`` must be ``step``.
            limit_train_batches (Optional[Union[int, float]]): Limits the number of train batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            limit_val_batches (Optional[Union[int, float]]): Limits the number of validation batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        self._check_step_epoch_mode(max_epochs, max_steps)
        self.limit_train_batches = limit_train_batches if isinstance(limit_train_batches, int) else float("inf")
        self.limit_val_batches = limit_val_batches if isinstance(limit_val_batches, int) else float("inf")
        self.validation_frequency = int(validation_frequency)
        super().__init__(fabric, compile, checkpoint_dir, validation_criteria, validation_unit, **kwargs)

    def _check_step_epoch_mode(
        self,
        max_epochs: Optional[Union[int, float]],
        max_steps: Optional[Union[int, float]],
    ) -> None:
        """Validate epoch- vs step-budget settings and assign ``self.max_epochs`` / ``self.max_steps``.

        Args:
            max_epochs (Optional[Union[int, float]]): Maximum epochs when not using ``max_steps``, or ``None`` in
                step-budget mode.
            max_steps (Optional[Union[int, float]]): Optimizer-step budget, or ``None`` in epoch-budget mode.
        """
        if max_steps is not None and max_epochs is not None:
            raise ValueError("Specify only one of max_epochs and max_steps, not both.")
        if max_steps is None and max_epochs is None:
            raise ValueError("Either max_epochs or max_steps must be set.")
        if max_steps is not None:
            self.max_steps = int(max_steps)
            self.max_epochs = None
        else:
            self.max_epochs = int(max_epochs)  # type: ignore[arg-type]
            self.max_steps = None

    def fit(
        self,
        model: SupervisedLightningModule,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        checkpoint: Optional[str] = None,
        load_only_model: bool = False,
    ) -> None:
        """Fit the model.

        Args:
            model (SupervisedLightningModule): The model to be trained. Should be a LightningModule instance.
            train_dataset (Dataset): The dataset for the training data.
            val_dataset (Optional[Dataset]): The dataset for the validation data. If not provided,
                validation will be skipped.
            checkpoint (Optional[str], optional): Path to a specific checkpoint to load or directory to load the latest
                checkpoint from. If provided, this will override automatic loading from the checkpoint directory.
            load_only_model (bool): If True, only the model is loaded from the checkpoint and the rest is ignored.
                This is useful when continuing training from a checkpoint without the optimizer and lr scheduler.
        """
        # Setup dataloaders.
        train_loader = self.fabric.setup_dataloaders(model.training_dataloader(train_dataset))
        if val_dataset is not None:
            val_loader = self.fabric.setup_dataloaders(model.validation_dataloader(val_dataset))
        else:
            val_loader = None

        # Setup optimizer and scheduler.
        optimizer, lr_scheduler = model.configure_optimizers()

        # Setup model and optimizer.
        model.training_step = torch.compile(model.training_step, mode="max-autotune", disable=not self.compile)
        model.validation_step = torch.compile(model.validation_step, mode="max-autotune", disable=not self.compile)
        model.forward = torch.compile(model.forward, mode="max-autotune", disable=not self.compile)
        model, optimizer = self.fabric.setup(model, optimizer)

        # Assemble checkpoint state (current epoch and global step will be added in save).
        state = {"model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}

        # Load last checkpoint if available.
        self.global_step, self.global_epoch, self.global_iteration = self.load_checkpoint(
            self.fabric, state, self.checkpoint_dir, checkpoint, load_only_model
        )

        self.should_stop = False

        if self.max_steps is not None:
            if self.global_step < self.max_steps:
                self.fit_step(
                    model,
                    optimizer,
                    train_loader,  # type: ignore[arg-type]
                    val_loader,  # type: ignore[arg-type]
                    lr_scheduler,
                    state,
                    val_dataset,
                )
        else:
            if self.global_epoch < self.max_epochs:  # type: ignore[operator]
                self.fit_epoch(
                    model,
                    optimizer,
                    train_loader,  # type: ignore[arg-type]
                    val_loader,  # type: ignore[arg-type]
                    lr_scheduler,
                    state,
                    val_dataset,
                )

        # Reset for next fit call.
        self.should_stop = False
        self.global_step = 0
        self.global_epoch = 0
        self._current_val_loss = float("inf") if self.validation_criteria == "min" else -float("inf")
        self._prev_val_loss = float("inf") if self.validation_criteria == "min" else -float("inf")
        self._current_train_loss = float("inf") if self.validation_criteria == "min" else -float("inf")
        self._prev_train_loss = float("inf") if self.validation_criteria == "min" else -float("inf")

    def fit_step(
        self,
        model: SupervisedLightningModule,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        lr_scheduler: Optional[LRScheduler],
        state: Dict[str, Union[nn.Module, Optimizer, Any]],
        val_dataset: Optional[Dataset],
    ) -> None:
        """The outer training loop for step-budget mode, running until ``global_step`` reaches ``max_steps``.

        Called from :meth:`fit` after dataloaders and the checkpoint state are configured.

        Args:
            model (SupervisedLightningModule): The model to be trained.
            optimizer (Optimizer): The optimizer for updating the model parameters.
            train_loader (DataLoader): The DataLoader providing the training batches.
            val_loader (Optional[DataLoader]): The DataLoader providing the validation batches. If not provided,
                validation will be skipped.
            lr_scheduler (Optional[LRScheduler]): The learning rate scheduler. If not provided, a constant
                learning rate will be used.
            state (Dict[str, Union[nn.Module, Optimizer, Any]]): A dictionary containing model, optimizer,
                and lr scheduler.
            val_dataset (Optional[Dataset]): The validation dataset. If provided, validation is used for checkpoint
                selection in :meth:`save_best`.
        """
        assert self.max_steps is not None
        trainer_pbar = self.tqdm_wrapper(
            range(self.global_step, self.max_steps),
            initial=self.global_step,
            total=self.max_steps,
            position=0,
            desc="Trainer",
        )
        while self.global_step < self.max_steps:
            self._log({"trainer/epoch": self.global_epoch})
            step_before = self.global_step
            self.train_loop(model, optimizer, train_loader, val_loader, lr_scheduler, state)
            steps_done = self.global_step - step_before
            self.fabric.call("train_loop_end", **rename_key(locals(), "self", "trainer"))

            if self.should_validate("epoch"):
                self.val_loop(model, val_loader)
                self.save_best(state, val_dataset is not None)
                self.fabric.call("val_loop_end", **rename_key(locals(), "self", "trainer"))

            self.global_epoch += 1
            metrics = {"train_loss": self._current_train_loss, "val_loss": self._current_val_loss}
            self._format_iterable(trainer_pbar, metrics)
            if isinstance(trainer_pbar, tqdm):
                trainer_pbar.update(steps_done)

            if self.should_stop:
                break

        self.close_tqdm_wrapper(trainer_pbar, self.max_steps)

    def fit_epoch(
        self,
        model: SupervisedLightningModule,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        lr_scheduler: Optional[LRScheduler],
        state: Dict[str, Union[nn.Module, Optimizer, Any]],
        val_dataset: Optional[Dataset],
    ) -> None:
        """The outer training loop for epoch-budget mode, running for up to ``max_epochs`` epochs.

        Called from :meth:`fit` after dataloaders and the checkpoint state are configured.

        Args:
            model (SupervisedLightningModule): The model to be trained.
            optimizer (Optimizer): The optimizer for updating the model parameters.
            train_loader (DataLoader): The DataLoader providing the training batches.
            val_loader (Optional[DataLoader]): The DataLoader providing the validation batches. If not provided,
                validation will be skipped.
            lr_scheduler (Optional[LRScheduler]): The learning rate scheduler. If not provided, a constant
                learning rate will be used.
            state (Dict[str, Union[nn.Module, Optimizer, Any]]): A dictionary containing model, optimizer,
                and lr scheduler.
            val_dataset (Optional[Dataset]): The validation dataset. If provided, validation is used for checkpoint
                selection in :meth:`save_best`.
        """
        assert self.max_epochs is not None
        trainer_pbar = self.tqdm_wrapper(
            range(self.global_epoch, self.max_epochs),
            initial=self.global_epoch,
            total=self.max_epochs,
            position=0,
            desc="Trainer",
        )

        for epoch in trainer_pbar:
            self._log({"trainer/epoch": self.global_epoch})
            self.train_loop(model, optimizer, train_loader, val_loader, lr_scheduler, state)
            self.fabric.call("train_loop_end", **rename_key(locals(), "self", "trainer"))

            if self.should_validate("epoch"):
                self.val_loop(model, val_loader)
                self.save_best(state, val_dataset is not None)
                self.fabric.call("val_loop_end", **rename_key(locals(), "self", "trainer"))

            self.global_epoch += 1
            metrics = {"train_loss": self._current_train_loss, "val_loss": self._current_val_loss}
            self._format_iterable(trainer_pbar, metrics)

            if self.global_epoch >= self.max_epochs:
                self.should_stop = True

        self.close_tqdm_wrapper(trainer_pbar, self.max_epochs)

    def train_loop(
        self,
        model: SupervisedLightningModule,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        state: Optional[Dict[str, Union[nn.Module, Optimizer, Any]]] = None,
    ) -> None:
        """The training loop running a single training epoch.

        Args:
            model (SupervisedLightningModule): The model to be trained.
            optimizer (Optimizer): The optimizer for updating the model parameters.
            train_loader (DataLoader): The DataLoader providing the training batches.
            val_loader (Optional[DataLoader]): The DataLoader providing the validation batches. If not provided,
                validation will be skipped.
            lr_scheduler (Optional[LRScheduler]): The learning rate scheduler. If not provided, a constant
                learning rate will be used.
            state (Optional[Dict[str, Union[nn.Module, Optimizer, Any]]]): A dictionary containing model, optimizer,
                and lr scheduler.
        """
        self.fabric.call("train_epoch_start", **rename_key(locals(), "self", "trainer"))
        model.train()
        train_steps = int(min(len(train_loader), self.limit_train_batches))
        train_pbar = self.tqdm_wrapper(train_loader, total=train_steps, desc="Training Loop", position=1, leave=False)
        train_loss, throughput = MeanMetric().to(self.fabric.device), MeanMetric().to(self.fabric.device)
        for batch_idx, batch in zip(range(train_steps), train_pbar):
            start_time = time()
            # Train one step.
            self.fabric.call("train_batch_start", **rename_key(locals(), "self", "trainer"))
            step_dict = model.training_step(batch, batch_idx=batch_idx, batch_len=train_steps)

            self.fabric.call("before_zero_grad", **rename_key(locals(), "self", "trainer"))
            optimizer.zero_grad()

            self.fabric.call("before_backward", **rename_key(locals(), "self", "trainer"))
            self.fabric.backward(step_dict["loss"])

            if model.max_grad_norm is not None:
                self.fabric.call("before_clip_grad", **rename_key(locals(), "self", "trainer"))
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.max_grad_norm)
                self._log({"train/total_grad_norm": total_grad_norm})

            self.fabric.call("before_optimizer_step", **rename_key(locals(), "self", "trainer"))
            optimizer.step()

            # Log metrics.
            self._log({f"train/{k}": v for k, v in step_dict.items()})
            self._log(
                {"trainer/global_step": self.global_step, "train/learning_rate": optimizer.param_groups[0]["lr"]}
            )

            if lr_scheduler is not None:
                self.fabric.call("before_step_scheduler", **rename_key(locals(), "self", "trainer"))
                lr_scheduler.step()

            # Update progress bar.
            train_loss(step_dict["loss"])
            self._format_iterable(train_pbar, train_loss.compute(), "train")

            # Log throughput.
            throughput(batch.shape[0] / (time() - start_time))
            self._log({"train/throughput": throughput.compute() * self.fabric.world_size})
            self.fabric.call("train_batch_end", **rename_key(locals(), "self", "trainer"))

            if self.should_validate("step"):
                self.val_loop(model, val_loader)
                self.save_best(state, val_loader is not None)
                self.fabric.call("val_loop_end", **rename_key(locals(), "self", "trainer"))
                model.train()

            self.global_step += 1
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        self._prev_train_loss = self._current_train_loss
        self._current_train_loss = train_loss.compute()
        self.close_tqdm_wrapper(train_pbar, train_steps)

    def val_loop(
        self,
        model: SupervisedLightningModule,
        val_loader: Optional[DataLoader],
    ) -> None:
        """The validation loop running a single validation epoch.

        Args:
            model (SupervisedLightningModule): The model to be evaluated.
            val_loader (DataLoader): The DataLoader providing the validation batches. If not provided,
                validation will be skipped.
        """
        # Skip validation if no validation DataLoader is provided.
        if val_loader is None:
            return

        self.fabric.call("val_epoch_start", **rename_key(locals(), "self", "trainer"))
        model.eval()
        mean_metrics, throughput = None, MeanMetric().to(self.fabric.device)

        val_steps = int(min(len(val_loader), self.limit_val_batches))
        val_pbar = self.tqdm_wrapper(val_loader, total=val_steps, desc="Validation Loop", position=1, leave=False)
        for batch_idx, batch in zip(range(val_steps), val_pbar):
            start_time = time()
            # Validate one step.
            self.fabric.call("validation_batch_start", **rename_key(locals(), "self", "trainer"))
            with torch.no_grad():
                step_dict = model.validation_step(batch, batch_idx, batch_len=val_steps)

            # Update mean metrics.
            if not mean_metrics:
                mean_metrics = {k: MeanMetric().to(self.fabric.device) for k in step_dict.keys()}
            for k, metric in mean_metrics.items():
                metric(step_dict[k])

            self._format_iterable(val_pbar, mean_metrics["loss"].compute(), "val")
            throughput(batch.shape[0] / (time() - start_time))
            self.fabric.call("val_batch_end", **rename_key(locals(), "self", "trainer"))

        # Log validation metrics.
        self._log({"val/throughput": throughput.compute() * self.fabric.world_size})
        assert mean_metrics is not None, "No validation metrics were computed."
        self._log({f"val/{k}": v.compute() for k, v in mean_metrics.items()})

        self.fabric.call("val_epoch_end", **rename_key(locals(), "self", "trainer"))
        assert mean_metrics is not None, "No validation metrics were computed."
        self._prev_val_loss = self._current_val_loss
        self._current_val_loss = mean_metrics["loss"].compute()
        self.close_tqdm_wrapper(val_pbar, val_steps)

    def should_validate(self, unit: Literal["epoch", "step"]) -> bool:
        """Whether to currently run validation.

        Args:
            unit (Literal["epoch", "step"]): The unit of the validation frequency.

        Returns:
            bool: Whether to currently run validation.
        """
        if unit == "epoch" and self.validation_unit == "epoch":
            return self.global_epoch % self.validation_frequency == 0
        elif unit == "step" and self.validation_unit == "step":
            return self.global_step % self.validation_frequency == 0
        else:
            return False
