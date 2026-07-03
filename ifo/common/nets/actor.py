from functools import partial
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    Normal,
    OneHotCategoricalStraightThrough,
    TanhTransform,
    TransformedDistribution,
)
from torch.distributions.utils import probs_to_logits

from ifo.common.distributions import (
    DictDistribution,
    MSEDistribution,
    MultiCategorical,
    MultiOneHotCategoricalStraightThrough,
)
from ifo.common.nets.dict import RemappingModuleDict
from ifo.common.utils.functions import orthogonal_init


class ActionHead(nn.Module):
    """Action head that automatically generates the correct distribution based on the action space.

    This module creates the appropriate linear layer and distribution for different action space types:
    - Box: Normal-family distributions (mean and log_std) or MSE distribution (mean only)
    - Discrete: Categorical distribution (outputs logits)
    - MultiDiscrete: MultiCategorical distribution (outputs logits for each discrete dimension)
    - Dict: DictDistribution (recursively handles nested action spaces)
    """

    box_head_distributions = {"normal", "tanh_normal", "scaled_normal", "mse"}
    categorical_head_distributions = {"categorical", "one_hot_categorical"}

    def __init__(
        self,
        in_features: int,
        action_space: spaces.Space,
        box_head_distribution: Literal["normal", "tanh_normal", "scaled_normal", "mse"] = "scaled_normal",
        categorical_head_distribution: Literal["categorical", "one_hot_categorical"] = "categorical",
        min_std: float = 0.01,
        max_std: float = 1.0,
        init_std: float = 0.0,
        mean_scale: float = 1.0,
        unimix: float = 0.0,
        ortho_init: bool = True,
    ) -> None:
        """Initialize the ActionHead.

        Args:
            in_features (int): Number of input features.
            action_space (spaces.Space): The gymnasium action space (Box, Discrete, MultiDiscrete, or Dict).
            box_head_distribution (Literal["normal", "tanh_normal", "scaled_normal", "mse"]): The distribution to
                use for the box head.
            categorical_head_distribution (Literal["categorical", "one_hot_categorical"]): The distribution to use
                for the categorical head.
            min_std (float): The minimum standard deviation for the actions (only used for scaled_normal, tanh_normal).
            max_std (float): The maximum standard deviation for the actions (only used for scaled_normal).
            init_std (float): The initial standard deviation for the actions (only used for scaled_normal, tanh_normal).
            mean_scale (float): The scale of the mean for the actions (only used for scaled_normal). E.g. if scale is 1,
                then the mean is in range [-1, 1], but if scale is 2, then the mean is in range [-2, 2].
            unimix (float): The percentage of uniform distribution to inject into the categorical
                distribution over actions, i.e. given some logits `l` and probabilities `p = softmax(l)`,
                then `p = (1 - self.unimix) * p + self.unimix * unif`,
                where `unif = 1 / self.discrete`.
            ortho_init (bool): Whether to use orthogonal initialization for the linear layers.
        """
        super().__init__()
        self.action_space = action_space
        self.min_std = min_std
        self.max_std = max_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.box_head_distribution = box_head_distribution
        self.categorical_head_distribution = categorical_head_distribution
        self.unimix = unimix

        assert (
            box_head_distribution in self.box_head_distributions
        ), f"Box head distribution must be one of: {self.box_head_distributions}"
        assert (
            categorical_head_distribution in self.categorical_head_distributions
        ), f"Categorical head distribution must be one of: {self.categorical_head_distributions}"

        gain = 0.0
        if isinstance(action_space, spaces.Box):
            self.action_dim = (
                int(np.prod(action_space.shape))
                if box_head_distribution == "mse"
                else int(np.prod(action_space.shape)) * 2
            )
            gain = 0.01
        elif isinstance(action_space, spaces.Discrete):
            self.action_dim = int(action_space.n)
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = list(action_space.nvec)
        elif isinstance(action_space, spaces.Dict):
            self.head = RemappingModuleDict(
                {
                    key: ActionHead(in_features, subspace, box_head_distribution, categorical_head_distribution)
                    for key, subspace in action_space.spaces.items()
                }
            )
            self.action_dim = {key: head.action_dim for key, head in self.head.items()}
        else:
            raise NotImplementedError(f"Action space {type(action_space)} is not supported.")
        self.head = nn.Linear(in_features, self.action_dim)
        self.apply(partial(orthogonal_init, gain=gain))

    def forward(self, x: Tensor) -> Distribution:
        """Forward pass that returns a distribution over actions.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Distribution: A distribution over actions appropriate for the action space.
        """
        if isinstance(self.action_space, spaces.Box):
            if self.box_head_distribution == "mse":
                mean = self.head(x)
                return MSEDistribution(mean, dims=len(self.action_space.shape), agg="mean")
            mean, std = torch.chunk(self.head(x), 2, dim=-1)
            if self.box_head_distribution == "normal":
                return Independent(Normal(mean, F.softplus(std)), 1)
            elif self.box_head_distribution == "tanh_normal":
                mean = 5 * torch.tanh(mean / 5)
                std = F.softplus(std + self.init_std) + self.min_std
                actions_dist = Normal(mean, std)
                actions_dist = Independent(TransformedDistribution(actions_dist, TanhTransform()), 1)
                return Independent(Normal(mean, F.softplus(std)), 1)
            elif self.box_head_distribution == "scaled_normal":
                std = (self.max_std - self.min_std) * torch.sigmoid(std + self.init_std) + self.min_std
                mean = self.mean_scale * torch.tanh(mean)
                return Independent(Normal(mean, std), 1)
        elif isinstance(self.action_space, spaces.Discrete):
            logits = self._uniform_mix(self.head(x))
            if self.categorical_head_distribution == "categorical":
                return Categorical(logits=logits)
            elif self.categorical_head_distribution == "one_hot_categorical":
                return OneHotCategoricalStraightThrough(logits=logits)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            logits = self.head(x)
            logits = [self._uniform_mix(logit) for logit in torch.split(logits, self.action_dim, dim=-1)]
            if self.categorical_head_distribution == "categorical":
                return MultiCategorical(logits=logits)
            elif self.categorical_head_distribution == "one_hot_categorical":
                return MultiOneHotCategoricalStraightThrough(logits=logits)
        elif isinstance(self.action_space, spaces.Dict):
            distributions = {key: action_head(x) for key, action_head in self.head.items()}
            return DictDistribution(distributions)
        else:
            raise NotImplementedError(f"Action space {type(self.action_space)} is not supported.")

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        """
        Apply uniform mixing to action logits to prevent overconfident predictions.

        This method injects a small amount of uniform distribution into the action
        distribution to improve exploration and prevent policy collapse.

        Args:
            logits (Tensor): Input action logits with shape [..., num_actions].

        Returns:
            Tensor: Mixed logits with uniform distribution injected.
        """
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        return logits
