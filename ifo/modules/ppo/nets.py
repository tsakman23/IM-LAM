from typing import Tuple

from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import Distribution

from ifo.common.utils.functions import merge_tc, orthogonal_init
from ifo.modules.bc.nets import ImpalaActorAgent


class ImpalaActorCriticAgent(ImpalaActorAgent):
    """
    Agent class with an actor and critic network and Impala feature extractor.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        channel_multiplier: int = 1,
    ) -> None:
        """
        Initializes ImpalaActorCriticAgent.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            action_space (spaces.Space): The action space of the environment.
            channel_multiplier (int, optional): Multiplier for the number of channels in the Impala feature extractor.
        """
        super().__init__(observation_space, action_space, channel_multiplier)
        self.critic = nn.Sequential(nn.ReLU(), nn.Linear(self.feature_extractor.out_features, 1))
        self.critic.apply(orthogonal_init)

    def forward(self, x: Tensor) -> Tuple[Distribution, Tensor]:
        """Forward pass.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tuple[Distribution, Tensor]: A tuple containing the action distribution and critic value.
        """
        x = merge_tc(x)
        x = self.feature_extractor(x)
        return self.actor(x), self.critic(x).squeeze(-1)
