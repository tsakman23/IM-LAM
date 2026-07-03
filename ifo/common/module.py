import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional

import hydra
from lightning import LightningModule
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ifo.common.utils.utility import tensordict_collate


class SupervisedLightningModule(LightningModule, ABC):
    """Lightning module for supervised learning."""

    def __init__(
        self,
        net: nn.Module,
        batch_size: int,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig] = None,
        num_workers: int = 0,
        max_grad_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Instantiate behavior cloning module

        Args:
            net (nn.Module): Policy neural network.
            batch_size (int): Batch size.
            optimizer (DictConfig): The optimizer config to use. Supports either:
                - Single learning rate: lr: 1e-4
                - Different learning rates for sub-networks: lr: {cnn: 1e-4, mlp: 2e-4, world_model.encoder: 3e-4}
            lr_scheduler (Optional[DictConfig]): The learning rate scheduler config to use.
                Applied to the single optimizer.
            num_workers (int, optional): Number of workers for dataloader.
            max_grad_norm (Optional[float], optional): Maximum gradient norm. If not provided,
                no clipping will be done.
        """
        super().__init__()
        self.net = net
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_grad_norm = max_grad_norm

        # Store optimizer and scheduler configs
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler

    @abstractmethod
    def training_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        """Train for one step.

        Args:
            batch (TensorDict): Batch with observation and action.
            batch_idx (int): Index of batch.
            batch_len (int): Length of batches in dataloader.
        Returns:
            TensorDict: Dictionary with loss and other metrics to log.
        """
        raise NotImplementedError

    def validation_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        """Validate for one step.

        Args:
            batch (TensorDict): Batch with observation and action.
            batch_idx (int): Index of batch.
            batch_len (int): Length of batches in dataloader.
        Returns:
            TensorDict: Dictionary with loss and other metrics to log.
        """
        raise NotImplementedError

    def _get_nested_attribute(self, obj: Any, attr_path: str) -> Any:
        """Get nested attribute from object using dot notation.

        Args:
            obj: The object to get the attribute from
            attr_path: Dot-separated attribute path (e.g., 'world_model.encoder')

        Returns:
            The nested attribute object

        Raises:
            AttributeError: If any part of the path doesn't exist
        """
        attrs = attr_path.split(".")
        current_obj = obj

        for attr in attrs:
            if not hasattr(current_obj, attr):
                obj_type = type(current_obj).__name__
                raise AttributeError(f"'{obj_type}' object has no attribute '{attr}' in path '{attr_path}'")
            current_obj = getattr(current_obj, attr)

        return current_obj

    def configure_optimizers(self) -> Any:
        """
        Configure single optimizer and optional learning rate scheduler.

        Supports two learning rate configurations:
        1. Single learning rate for the whole network
        2. Different learning rates for named sub-networks

        Returns:
            Tuple[Optimizer, Optional[LRScheduler]]:
                Single optimizer and optional scheduler (can be None).
        """
        # Check if lr is a dict of subnet-specific learning rates
        lr_config = self.optimizer_cfg.get("lr", None)

        if isinstance(lr_config, (dict, DictConfig)):
            # Different learning rates for different subnets
            param_groups = []

            for subnet_name, lr_value in lr_config.items():
                subnet_name = str(subnet_name)  # Ensure string type
                # Get the sub-network by name recursively (e.g., world_model.encoder)
                subnet = self._get_nested_attribute(self.net, subnet_name)
                if hasattr(subnet, "parameters"):
                    param_groups.append({"params": subnet.parameters(), "lr": lr_value})
                else:
                    warnings.warn(f"Subnet {subnet_name} has no parameters")

            # Create optimizer config without the lr field and instantiate with param_groups
            param_group_optimizer_cfg = deepcopy(self.optimizer_cfg)
            param_group_optimizer_cfg.pop("lr")

            optimizer = hydra.utils.instantiate(param_group_optimizer_cfg, _partial_=True)
            optimizer = optimizer(params=param_groups)
        else:
            # Single learning rate for the whole network
            optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.net.parameters())

        # Handle optional scheduler
        scheduler = None
        if self.lr_scheduler_cfg is not None:
            scheduler = hydra.utils.instantiate(self.lr_scheduler_cfg, optimizer=optimizer)

        return optimizer, scheduler

    def training_dataloader(self, dataset: Dataset) -> DataLoader:
        """
        Create training dataloader from dataset.

        Args:
            dataset (Dataset): Training dataset.

        Returns:
            DataLoader: Dataloader initialized with dataset.
        """
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # Shuffle not supported for IterableDataset.
            shuffle=False if isinstance(dataset, IterableDataset) else True,
            collate_fn=tensordict_collate,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )
        return dataloader

    def validation_dataloader(self, dataset: Dataset) -> DataLoader:
        """Create validation dataloader from dataset.

        Args:
            dataset (Dataset): Validation dataset

        Returns:
            DataLoader: Dataloader initialized with dataset.
        """
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # Shuffle not supported for IterableDataset.
            shuffle=False,
            collate_fn=tensordict_collate,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )
        return dataloader
