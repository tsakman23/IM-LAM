from typing import Dict, Optional

from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn
from torch.distributions import Distribution

from ifo.common.module import SupervisedLightningModule


class BehaviorCloningModule(SupervisedLightningModule):
    """Behavior cloning module."""

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
        super().__init__(
            net=net,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            num_workers=num_workers,
            max_grad_norm=max_grad_norm,
            **kwargs,
        )

    def training_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        action_distribution = self.net(batch["observation"])
        loss = -action_distribution.log_prob(batch["action"][:, -1]).mean()
        return TensorDict({"loss": loss})

    def validation_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        return self.training_step(batch, batch_idx, batch_len)

    def forward(self, batch: TensorDict) -> Dict[str, Distribution]:
        action_distribution = self.net(batch["observation"])
        return {"action_distribution": action_distribution}


class BehaviorCloningTransformerModule(BehaviorCloningModule):
    """Behavior cloning module with Transformer."""

    def training_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        action_distribution = self.net(batch["observation"])
        loss = -action_distribution.log_prob(batch["action"]).mean()
        return TensorDict({"loss": loss})

    def forward(self, batch: TensorDict) -> Dict[str, Distribution]:
        action_distribution = self.net.predict(batch["observation"])
        return {"action_distribution": action_distribution}
