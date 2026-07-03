from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import hydra
import numpy as np
import torch
from gymnasium import spaces
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from tensordict import TensorDict
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.distributions.utils import probs_to_logits

from ifo.common.nets.actor import ActionHead
from ifo.common.nets.base import CNN, MLP, DeCNN, LayerNormChannelLast, LayerNormGRUCell
from ifo.common.utils.functions import symlog
from ifo.common.utils.model import cnn_forward
from ifo.modules.dreamerv3.utils import compute_stochastic_state, init_weights, uniform_init_weights


class DreamerV3ActorCriticAgent(nn.Module):
    """
    Actor-critic agent for the Dreamer-V3 model.

    This class implements the complete DreamerV3 agent architecture, including the world model,
    actor, and critic networks. It handles both CNN and MLP encoders/decoders for different
    observation types and supports both continuous and discrete action spaces.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        world_model: Dict[str, Any],
        actor: Dict[str, Any],
        critic: Dict[str, Any],
        normalizer: Dict[str, Any],
        hafner_initialization: bool = False,
    ) -> None:
        """
        Initialize the DreamerV3 Actor-Critic Agent.

        Args:
            observation_space (spaces.Dict): The observation space from the environment.
            action_space (spaces.Space): The action space from the environment.
            world_model (Dict[str, Any]): Configuration dictionary for the world model components.
            actor (Dict[str, Any]): Configuration dictionary for the actor network.
            critic (Dict[str, Any]): Configuration dictionary for the critic network.
            normalizer (Dict[str, Any]): Configuration dictionary for the normalizer.
            hafner_initialization (bool, optional): Whether to use Hafner initialization scheme. Defaults to False.
        """
        super().__init__()
        if isinstance(action_space, spaces.Discrete):
            self.actions_dim = action_space.n
        elif isinstance(action_space, spaces.Box):
            self.actions_dim = int(sum(action_space.shape))
        else:
            raise NotImplementedError(f"Action space {type(action_space)} is not supported.")

        # Compute sizes
        self.recurrent_state_size = world_model["recurrent_model"]["recurrent_state_size"]
        self.stochastic_size = world_model["stochastic_size"]
        self.discrete_size = world_model["discrete_size"]
        self.stochastic_state_size = self.stochastic_size * self.discrete_size
        self.latent_state_size = self.stochastic_state_size + self.recurrent_state_size

        # Get observation keys
        self.cnn_keys: List[str] = world_model["cnn_keys"]["encoder"]
        self.mlp_keys: List[str] = world_model["mlp_keys"]["encoder"]
        self.obs_keys: List[str] = self.cnn_keys + self.mlp_keys
        assert self.obs_keys, "Observation keys must be provided"

        # Build encoder for world model
        cnn_encoder = None
        if self.cnn_keys:
            cnn_input_channels = [int(np.prod(observation_space[cnn_key].shape[-3])) for cnn_key in self.cnn_keys]
            cnn_image_size = observation_space[self.cnn_keys[0]].shape[-2:]
            cnn_encoder = CNNEncoder(
                keys=self.cnn_keys,
                input_channels=cnn_input_channels,
                image_size=cnn_image_size,
                channels_multiplier=world_model["encoder"]["cnn_channels_multiplier"],
                layer_norm_cls=hydra.utils.get_class(world_model["encoder"]["cnn_layer_norm"]["cls"]),
                layer_norm_kw=world_model["encoder"]["cnn_layer_norm"]["kw"],
                layer_args=world_model["encoder"]["cnn_layer_args"],
                activation=hydra.utils.get_class(world_model["encoder"]["cnn_act"]),
            )

        mlp_encoder = None
        if self.mlp_keys:
            mlp_input_dims = [observation_space[mlp_key].shape[-1] for mlp_key in self.cnn_keys]
            mlp_encoder = MLPEncoder(
                keys=self.mlp_keys,
                input_dims=mlp_input_dims,
                mlp_layers=world_model["encoder"]["mlp_layers"],
                dense_units=world_model["encoder"]["dense_units"],
                activation=hydra.utils.get_class(world_model["encoder"]["dense_act"]),
                layer_norm_cls=hydra.utils.get_class(world_model["encoder"]["mlp_layer_norm"]["cls"]),
                layer_norm_kw=world_model["encoder"]["mlp_layer_norm"]["kw"],
                layer_args=world_model["encoder"]["mlp_layer_args"],
            )
        encoder = MultiEncoder(cnn_encoder, mlp_encoder)

        # Build decoder for world model
        cnn_decoder = None
        if self.cnn_keys:
            cnn_decoder = CNNDecoder(
                keys=self.cnn_keys,
                output_channels=cnn_input_channels,
                channels_multiplier=world_model["decoder"]["cnn_channels_multiplier"],
                latent_state_size=self.latent_state_size,
                cnn_encoder_output_shape=cnn_encoder.output_shape,
                image_size=cnn_image_size,
                activation=hydra.utils.get_class(world_model["decoder"]["cnn_act"]),
                layer_norm_cls=hydra.utils.get_class(world_model["decoder"]["cnn_layer_norm"]["cls"]),
                layer_norm_kw=world_model["decoder"]["cnn_layer_norm"]["kw"],
                layer_args=world_model["decoder"]["cnn_layer_args"],
                last_layer_args=world_model["decoder"]["cnn_last_layer_args"],
            )

        mlp_decoder = None
        if self.mlp_keys:
            mlp_decoder = MLPDecoder(
                keys=self.mlp_keys,
                output_dims=mlp_input_dims,
                latent_state_size=self.latent_state_size,
                mlp_layers=world_model["decoder"]["mlp_layers"],
                dense_units=world_model["decoder"]["dense_units"],
                activation=hydra.utils.get_class(world_model["decoder"]["dense_act"]),
                layer_norm_cls=hydra.utils.get_class(world_model["decoder"]["mlp_layer_norm"]["cls"]),
                layer_norm_kw=world_model["decoder"]["mlp_layer_norm"]["kw"],
                layer_args=world_model["decoder"]["mlp_layer_args"],
            )

        decoder = MultiDecoder(cnn_decoder, mlp_decoder)

        # Build reward head for world model
        reward_ln_cls = hydra.utils.get_class(world_model["reward_model"]["layer_norm"]["cls"])
        reward_model = MLP(
            input_dims=self.latent_state_size,
            output_dim=world_model["reward_model"]["bins"],
            hidden_sizes=[world_model["reward_model"]["dense_units"]] * world_model["reward_model"]["mlp_layers"],
            activation=hydra.utils.get_class(world_model["reward_model"]["dense_act"]),
            flatten_dim=None,
            norm_layer=reward_ln_cls,
            norm_args={
                **world_model["reward_model"]["layer_norm"]["kw"],
                "normalized_shape": world_model["reward_model"]["dense_units"],
            },
        )

        # Build continue head for world model
        discount_ln_cls = hydra.utils.get_class(world_model["discount_model"]["layer_norm"]["cls"])
        continue_model = MLP(
            input_dims=self.latent_state_size,
            output_dim=1,
            hidden_sizes=[world_model["discount_model"]["dense_units"]] * world_model["discount_model"]["mlp_layers"],
            activation=hydra.utils.get_class(world_model["discount_model"]["dense_act"]),
            flatten_dim=None,
            norm_layer=discount_ln_cls,
            norm_args={
                **world_model["discount_model"]["layer_norm"]["kw"],
                "normalized_shape": world_model["discount_model"]["dense_units"],
            },
        )

        # Build world model
        if world_model["recurrent_model"]["block_diagonal"]:
            recurrent_model = BlockDiagonalRecurrentModel(
                input_size=self.actions_dim + self.stochastic_state_size,
                recurrent_state_size=world_model["recurrent_model"]["recurrent_state_size"],
                hidden_size=world_model["recurrent_model"]["hidden_size"],
                blocks=world_model["recurrent_model"]["blocks"],
                layer_norm_cls=hydra.utils.get_class(world_model["recurrent_model"]["layer_norm"]["cls"]),
                layer_norm_kw=world_model["recurrent_model"]["layer_norm"]["kw"],
            )
        else:
            recurrent_model = RecurrentModel(
                input_size=self.actions_dim + self.stochastic_state_size,
                recurrent_state_size=world_model["recurrent_model"]["recurrent_state_size"],
                dense_units=[world_model["recurrent_model"]["dense_units"]]
                * world_model["recurrent_model"]["mlp_layers"],
                layer_norm_cls=hydra.utils.get_class(world_model["recurrent_model"]["layer_norm"]["cls"]),
                layer_norm_kw=world_model["recurrent_model"]["layer_norm"]["kw"],
            )

        represention_model_input_size = encoder.output_dim + self.recurrent_state_size
        representation_ln_cls = hydra.utils.get_class(world_model["representation_model"]["layer_norm"]["cls"])
        representation_model = MLP(
            input_dims=represention_model_input_size,
            output_dim=self.stochastic_state_size,
            hidden_sizes=[world_model["representation_model"]["hidden_size"]]
            * world_model["representation_model"]["mlp_layers"],
            activation=hydra.utils.get_class(world_model["representation_model"]["dense_act"]),
            flatten_dim=None,
            norm_layer=representation_ln_cls,
            norm_args={
                **world_model["representation_model"]["layer_norm"]["kw"],
                "normalized_shape": world_model["representation_model"]["hidden_size"],
            },
        )
        transition_ln_cls = hydra.utils.get_class(world_model["transition_model"]["layer_norm"]["cls"])
        transition_model = MLP(
            input_dims=self.recurrent_state_size,
            output_dim=self.stochastic_state_size,
            hidden_sizes=[world_model["transition_model"]["hidden_size"]]
            * world_model["transition_model"]["mlp_layers"],
            activation=hydra.utils.get_class(world_model["transition_model"]["dense_act"]),
            flatten_dim=None,
            norm_layer=transition_ln_cls,
            norm_args={
                **world_model["transition_model"]["layer_norm"]["kw"],
                "normalized_shape": world_model["transition_model"]["hidden_size"],
            },
        )

        rssm = RSSM(
            recurrent_model=recurrent_model.apply(init_weights),
            representation_model=representation_model.apply(init_weights),
            transition_model=transition_model.apply(init_weights),
            discrete=world_model["discrete_size"],
            unimix=world_model["unimix"],
            learnable_initial_recurrent_state=world_model["learnable_initial_recurrent_state"],
        )

        self.world_model = WorldModel(
            encoder=encoder.apply(init_weights),
            rssm=rssm,
            decoder=decoder.apply(init_weights),
            reward_model=reward_model.apply(init_weights),
            continue_model=continue_model.apply(init_weights),
        )

        # Build actor model
        self.actor = Actor(
            action_space=action_space,
            latent_state_size=self.latent_state_size,
            dense_units=actor["dense_units"],
            activation=hydra.utils.get_class(actor["dense_act"]),
            mlp_layers=actor["mlp_layers"],
            layer_norm_cls=hydra.utils.get_class(actor["layer_norm"]["cls"]),
            layer_norm_kw=actor["layer_norm"]["kw"],
            box_head_distribution=actor["box_head_distribution"],
            categorical_head_distribution=actor["categorical_head_distribution"],
            min_std=actor["min_std"],
            max_std=actor["max_std"],
            init_std=actor["init_std"],
            mean_scale=actor["mean_scale"],
            unimix=actor["unimix"],
        )

        # Build critic model
        critic_ln_cls = hydra.utils.get_class(critic["layer_norm"]["cls"])
        self.critic = MLP(
            input_dims=self.latent_state_size,
            output_dim=critic["bins"],
            hidden_sizes=[critic["dense_units"]] * critic["mlp_layers"],
            activation=hydra.utils.get_class(critic["dense_act"]),
            flatten_dim=None,
            norm_layer=critic_ln_cls,
            norm_args={
                **critic["layer_norm"]["kw"],
                "normalized_shape": critic["dense_units"],
            },
        )

        # Initialize parameters
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.world_model.encoder.apply(init_weights)
        self.world_model.decoder.apply(init_weights)
        self.world_model.reward_model.apply(init_weights)
        self.world_model.continue_model.apply(init_weights)

        # Initialize according to Hafner if specified
        if hafner_initialization:
            self.actor.mlp_head.apply(uniform_init_weights(0.01))
            self.critic.model[-1].apply(uniform_init_weights(0.0))
            self.world_model.rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))
            self.world_model.rssm.representation_model.model[-1].apply(uniform_init_weights(1.0))
            self.world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
            self.world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))
            if self.world_model.decoder.mlp_decoder is not None:
                self.world_model.decoder.mlp_decoder.heads.apply(uniform_init_weights(1.0))
            if self.world_model.decoder.cnn_decoder is not None:
                self.world_model.decoder.cnn_decoder.model[-1].model[-1].apply(uniform_init_weights(1.0))

        # Build target critic model
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.requires_grad_(False)

        # Build normalizer model
        self.return_normalizer = Normalize(**normalizer["return"])
        self.value_normalizer = Normalize(**normalizer["value"])
        self.advantage_normalizer = Normalize(**normalizer["advantage"])

    @torch.no_grad()
    def init_states(self, dones: np.ndarray) -> Tuple[Tensor, Tensor, Tensor]:
        """Return initial dreamer state.

        Args:
            dones (np.ndarray): which environments' states to reset.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Initial recurrent state, stochastic state and actions.
        """
        num_envs = len(dones)
        recurrent_state, stochastic_state = self.world_model.rssm.get_initial_states((1, num_envs))
        stochastic_state = stochastic_state.reshape(1, num_envs, -1)
        actions = torch.zeros(1, num_envs, self.actions_dim, device=recurrent_state.device)
        dreamer_state = (recurrent_state, stochastic_state, actions)
        return dreamer_state

    @torch.no_grad()
    def update_states(
        self, dones: np.ndarray, dreamer_state: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Initialize the states and the actions for the ended environments.

        Args:
            dones (np.ndarray): which environments' states to reset.
            dreamer_state (Tuple[Tensor, Tensor, Tensor]): Dreamer state to update.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Updated dreamer state.
        """
        num_envs = len(dones)
        recurrent_state, stochastic_state, actions = dreamer_state
        if dones.any():
            new_recurrent_state, new_stochastic_state = self.world_model.rssm.get_initial_states((1, num_envs))
            recurrent_state[:, dones] = new_recurrent_state.reshape(1, num_envs, -1)[:, dones]
            stochastic_state[:, dones] = new_stochastic_state.reshape(1, num_envs, -1)[:, dones]
            actions[:, dones] = torch.zeros_like(actions[:, dones])
        # Clone the states to avoid modifying the original states and avoiding inplace operations with torch.compile
        new_dreamer_state = (recurrent_state.clone(), stochastic_state.clone(), actions.clone())
        return new_dreamer_state

    def forward(
        self, obs: TensorDict, dreamer_state: Tuple[Tensor, Tensor, Tensor], greedy: bool = False
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Return the greedy actions.

        Args:
            obs (TensorDict): the current observations.
            dreamer_state (Tuple[Tensor, Tensor, Tensor]): Dreamer state.
            greedy (bool): whether or not to sample the actions.
                Default to False.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]: Actions and dreamer state.
        """
        recurrent_state, stochastic_state, actions = dreamer_state
        embedded_obs = self.world_model.encoder(obs)
        new_recurrent_state = self.world_model.rssm.recurrent_model(
            torch.cat((stochastic_state, actions), -1), recurrent_state
        )
        _, new_stochastic_state = self.world_model.rssm._representation(new_recurrent_state, embedded_obs[None])
        new_stochastic_state = new_stochastic_state.view(
            *new_stochastic_state.shape[:-2], self.stochastic_size * self.discrete_size
        )
        actions_dist = self.actor(torch.cat((new_stochastic_state, new_recurrent_state), -1))
        new_actions = actions_dist.mode if greedy else actions_dist.rsample()

        new_dreamer_state = (new_recurrent_state, new_stochastic_state, new_actions)

        return new_actions[0], new_dreamer_state


class WorldModel(nn.Module):
    """
    Wrapper class for the World model.
    """

    def __init__(
        self,
        encoder: MultiEncoder,
        rssm: RSSM,
        decoder: MultiDecoder,
        reward_model: MLP,
        continue_model: MLP,
    ) -> None:
        """
        Initialize the WorldModel.

        Args:
            encoder (MultiEncoder): The encoder for processing observations.
            rssm (RSSM): The Recurrent State-Space Model.
            decoder (MultiDecoder): The decoder for reconstructing observations.
            reward_model (MLP): The reward prediction model.
            continue_model (MLP): The continue/termination prediction model.
        """
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.reward_model = reward_model
        self.continue_model = continue_model


class CNNEncoder(nn.Module):
    """The Dreamer-V3 image encoder.

    This is composed of 4 `nn.Conv2d` with kernel_size=3, stride=2 and padding=1.
    No bias is used if a `nn.LayerNorm` is used after the convolution.
    If more than one image is to be encoded, then those will be concatenated on the
    channel dimension and fed to the encoder.
    """

    def __init__(
        self,
        keys: Sequence[str],
        input_channels: Sequence[int],
        image_size: Tuple[int, int],
        channels_multiplier: int,
        base_multipliers: Sequence[int] = [2, 3, 4, 4],
        layer_norm_cls: Callable[..., nn.Module] = LayerNormChannelLast,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        layer_args: Dict[str, Any] = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": False},
        activation: nn.Module = nn.SiLU,
    ) -> None:
        """
        Initialize the CNN encoder.

        Args:
            keys (Sequence[str]): The keys representing the image observations to encode.
            input_channels (Sequence[int]): The input channels, one for each image observation to encode.
            image_size (Tuple[int, int]): The image size as (Height, Width).
            channels_multiplier (int): The multiplier for the output channels. Given the 4 stages,
                the 4 output channels will be [1, 2, 4, 8] * `channels_multiplier`.
            base_multipliers (Sequence[int]): The base multipliers for the output channels.
                Defaults to [2, 3, 4, 4].
            layer_norm_cls (Callable[..., nn.Module]): The layer norm to apply after the input projection.
                Defaults to LayerNormChannelLast.
            layer_norm_kw (Dict[str, Any]): The kwargs of the layer norm.
                Defaults to {"eps": 1e-3}.
            layer_args (Dict[str, Any]): The kwargs of the convolutional layers.
                Defaults to {"kernel_size": 4, "stride": 2, "padding": 1, "bias": False}.
            activation (nn.Module): The activation function. Defaults to nn.SiLU.
            stages (int): How many stages for the CNN. Defaults to 4.
        """
        super().__init__()
        self.keys = keys
        self.input_dim = (sum(input_channels), *image_size)
        hidden_channels = [i * channels_multiplier for i in base_multipliers]
        self.model = nn.Sequential(
            CNN(
                input_channels=self.input_dim[0],
                hidden_channels=hidden_channels,
                cnn_layer=nn.Conv2d,
                layer_args=layer_args,
                activation=activation,
                norm_layer=[layer_norm_cls] * len(hidden_channels),
                norm_args=[{**layer_norm_kw, "normalized_shape": i} for i in hidden_channels],
            ),
            nn.Flatten(-3, -1),
        )
        self.output_shape = (
            hidden_channels[-1],
            image_size[0] // (2 ** len(base_multipliers)),
            image_size[1] // (2 ** len(base_multipliers)),
        )
        self.output_dim = math.prod(self.output_shape)

    def forward(self, obs: TensorDict) -> Tensor:
        """
        Forward pass through the CNN encoder.

        Args:
            obs (TensorDict): Dictionary of observations with string keys.
                Each tensor should have shape matching the expected input dimensions.

        Returns:
            Tensor: Encoded features with shape [batch_size, output_dim].
        """
        x = torch.cat([obs[k] for k in self.keys], dim=-3)  # channels dimension
        return cnn_forward(self.model, x, x.shape[-3:], (-1,))


class MLPEncoder(nn.Module):
    """The Dreamer-V3 vector encoder.

    This is composed of N `nn.Linear` layers, where N is specified by `mlp_layers`.
    No bias is used if a `nn.LayerNorm` is used after the linear layer. If more than
    one vector is to be encoded, then those will concatenated on the last dimension
    before being fed to the encoder.
    """

    def __init__(
        self,
        keys: Sequence[str],
        input_dims: Sequence[int],
        mlp_layers: int = 4,
        dense_units: int = 512,
        layer_norm_cls: Callable[..., nn.Module] = nn.LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        layer_args: Dict[str, Any] = {"bias": False},
        activation: nn.Module = nn.SiLU,
        symlog_inputs: bool = True,
    ) -> None:
        """
        Initialize the MLP encoder.

        Args:
            keys (Sequence[str]): The keys representing the vector observations to encode.
            input_dims (Sequence[int]): The dimensions of every vector to encode.
            mlp_layers (int): How many MLP layers. Defaults to 4.
            dense_units (int): The dimension of every MLP. Defaults to 512.
            layer_norm_cls (Callable[..., nn.Module]): The layer norm to apply after the input projection.
                Defaults to LayerNorm.
            layer_norm_kw (Dict[str, Any]): The kwargs of the layer norm.
                Defaults to {"eps": 1e-3}.
            layer_args (Dict[str, Any]): The kwargs of the linear layers.
                Defaults to {"bias": False}.
            activation (nn.Module): The activation function after every layer.
                Defaults to nn.SiLU.
            symlog_inputs (bool): Whether to squash the input with the symlog function.
                Defaults to True.
        """
        super().__init__()
        self.keys = keys
        self.input_dim = sum(input_dims)
        self.model = MLP(
            self.input_dim,
            None,
            [dense_units] * mlp_layers,
            activation=activation,
            layer_args=layer_args,
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        self.output_dim = dense_units
        self.symlog_inputs = symlog_inputs

    def forward(self, obs: TensorDict) -> Tensor:
        """
        Forward pass through the MLP encoder.

        Args:
            obs (TensorDict): Dictionary of vector observations with string keys.
                Each tensor should have shape [batch_size, feature_dim].

        Returns:
            Tensor: Encoded features with shape [batch_size, output_dim].
        """
        x = torch.cat([symlog(obs[k]) if self.symlog_inputs else obs[k] for k in self.keys], -1)
        return self.model(x)


class CNNDecoder(nn.Module):
    """The exact inverse of the `CNNEncoder` class.

    It reconstructs the observation in the inverse order of the encoder.
    If multiple images are to be reconstructed, then it will create a
    dictionary with an entry for every reconstructed image. No bias is used if a
    `nn.LayerNorm` is used after the `nn.Conv2dTranspose` layer.
    """

    def __init__(
        self,
        keys: Sequence[str],
        output_channels: Sequence[int],
        channels_multiplier: int,
        latent_state_size: int,
        cnn_encoder_output_shape: Tuple[int, int, int],
        image_size: Tuple[int, int],
        base_multipliers: Sequence[int] = [2, 3, 4, 4],
        activation: nn.Module = nn.SiLU,
        layer_norm_cls: Callable[..., nn.Module] = LayerNormChannelLast,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        layer_args: Dict[str, Any] = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": False},
        last_layer_args: Dict[str, Any] = {"kernel_size": 3, "stride": 2, "padding": 1},
        stages: int = 4,
    ) -> None:
        """
        Initialize the CNN decoder.

        Args:
            keys (Sequence[str]): The keys of the image observation to be reconstructed.
            output_channels (Sequence[int]): The output channels, one for every image observation.
            channels_multiplier (int): The channels multiplier, same for the encoder network.
            latent_state_size (int): The size of the latent state. Before applying the decoder,
                a `nn.Linear` layer is used to project the latent state to a feature vector
                of dimension [8 * `channels_multiplier`, 4, 4].
            cnn_encoder_output_shape (Tuple[int, int, int]): The output shape of the image encoder in the
                form of (channels, height, width).
            image_size (Tuple[int, int]): The final image size.
            base_multipliers (Sequence[int]): The base multipliers for the output channels.
                Defaults to [2, 3, 4, 4].
            activation (nn.Module): The activation function. Defaults to nn.SiLU.
            layer_norm_cls (Callable[..., nn.Module]): The layer norm to apply after the input projection.
                Defaults to LayerNormChannelLast.
            layer_norm_kw (Dict[str, Any]): The kwargs of the layer norm.
                Defaults to {"eps": 1e-3}.
            layer_args (Dict[str, Any]): The kwargs of the convolutional layers.
                Defaults to {"kernel_size": 4, "stride": 2, "padding": 1, "bias": False}.
            last_layer_args (Dict[str, Any]): The kwargs of the last convolutional layer.
                Defaults to {"kernel_size": 4, "stride": 2, "padding": 1}.
            stages (int): How many stages in the CNN decoder. Defaults to 4.
        """
        super().__init__()
        self.keys = keys
        self.output_channels = output_channels
        self.cnn_encoder_output_shape = cnn_encoder_output_shape
        self.image_size = image_size
        self.output_dim = (sum(output_channels), *image_size)
        hidden_channels = [i * channels_multiplier for i in reversed(base_multipliers)]
        self.model = nn.Sequential(
            nn.Linear(latent_state_size, math.prod(self.cnn_encoder_output_shape)),
            nn.Unflatten(1, self.cnn_encoder_output_shape),
            DeCNN(
                input_channels=self.cnn_encoder_output_shape[0],
                hidden_channels=hidden_channels[:-1] + [self.output_dim[0]],
                cnn_layer=nn.ConvTranspose2d,
                layer_args=[layer_args for _ in range(len(hidden_channels) - 1)] + [last_layer_args],
                activation=[activation for _ in range(len(hidden_channels) - 1)] + [None],
                norm_layer=[layer_norm_cls for _ in range(len(hidden_channels) - 1)] + [None],
                norm_args=(
                    [{**layer_norm_kw, "normalized_shape": hidden_channels[i]} for i in range(len(hidden_channels) - 1)]
                    + [None]
                ),
            ),
        )

    def forward(self, latent_states: Tensor) -> TensorDict:
        """
        Forward pass through the CNN decoder.

        Args:
            latent_states (Tensor): Latent state tensor with shape [batch_size, latent_state_size].

        Returns:
            TensorDict: Dictionary of reconstructed image observations.
                Each tensor has shape [batch_size, channels, height, width].
        """
        cnn_out = cnn_forward(self.model, latent_states, (latent_states.shape[-1],), self.output_dim)
        return TensorDict(
            {k: rec_obs for k, rec_obs in zip(self.keys, torch.split(cnn_out, self.output_channels, -3))},
            batch_size=latent_states.shape[:-1],
            device=latent_states.device,
        )


class MLPDecoder(nn.Module):
    """The exact inverse of the MLPEncoder.

    This is composed of N `nn.Linear` layers, where N is specified by `mlp_layers`.
    No bias is used if a `nn.LayerNorm` is used after the linear layer. If more than
    one vector is to be decoded, then it will create a dictionary with an entry for
    every reconstructed vector.
    """

    def __init__(
        self,
        keys: Sequence[str],
        output_dims: Sequence[int],
        latent_state_size: int,
        mlp_layers: int = 4,
        dense_units: int = 512,
        activation: nn.Module = nn.SiLU,
        layer_norm_cls: Callable[..., nn.Module] = nn.LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        layer_args: Dict[str, Any] = {"bias": False},
    ) -> None:
        """
        Initialize the MLP decoder.

        Args:
            keys (Sequence[str]): The keys representing the vector observations to decode.
            output_dims (Sequence[int]): The dimensions of every vector to decode.
            latent_state_size (int): The dimension of the latent state.
            mlp_layers (int): How many MLP layers. Defaults to 4.
            dense_units (int): The dimension of every MLP. Defaults to 512.
            activation (nn.Module): The activation function after every layer.
                Defaults to nn.SiLU.
            layer_norm_cls (Callable[..., nn.Module]): The layer norm to apply after the input projection.
                Defaults to LayerNorm.
            layer_norm_kw (Dict[str, Any]): The kwargs of the layer norm.
                Defaults to {"eps": 1e-3}.
            layer_args (Dict[str, Any]): The kwargs of the linear layers.
                Defaults to {"bias": False}.
        """
        super().__init__()
        self.output_dims = output_dims
        self.keys = keys
        self.model = MLP(
            latent_state_size,
            None,
            [dense_units] * mlp_layers,
            activation=activation,
            layer_args=layer_args,
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        self.heads = nn.ModuleList([nn.Linear(dense_units, mlp_dim) for mlp_dim in self.output_dims])

    def forward(self, latent_states: Tensor) -> TensorDict:
        """
        Forward pass through the MLP decoder.

        Args:
            latent_states (Tensor): Latent state tensor with shape [batch_size, latent_state_size].

        Returns:
            TensorDict: Dictionary of reconstructed vector observations.
                Each tensor has shape [batch_size, feature_dim].
        """
        x = self.model(latent_states)
        return TensorDict(
            {k: h(x) for k, h in zip(self.keys, self.heads)},
            batch_size=latent_states.shape[:-1],
            device=latent_states.device,
        )


class RecurrentModel(nn.Module):
    """Recurrent model for the model-base Dreamer-V3 agent.

    This implementation uses the `sheeprl.models.models.LayerNormGRUCell`, which combines
    the standard GRUCell from PyTorch with the `nn.LayerNorm`, where the normalization is applied
    right after having computed the projection from the input to the weight space.
    """

    def __init__(
        self,
        input_size: int,
        recurrent_state_size: int,
        dense_units: Sequence[int],
        activation_fn: nn.Module = nn.SiLU,
        layer_norm_cls: Callable[..., nn.Module] = nn.LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
    ) -> None:
        """
        Initialize the RecurrentModel.

        Args:
            input_size (int): The input size of the model.
            recurrent_state_size (int): The size of the recurrent state.
            dense_units (Sequence[int]): The number of dense units.
            activation_fn (nn.Module): The activation function. Defaults to SiLU.
            layer_norm_cls (Callable[..., nn.Module]): The layer norm to apply after the input projection.
                Defaults to LayerNorm.
            layer_norm_kw (Dict[str, Any]): The kwargs of the layer norm.
                Defaults to {"eps": 1e-3}.
        """
        super().__init__()
        self.mlp = MLP(
            input_dims=input_size,
            output_dim=None,
            hidden_sizes=dense_units,
            activation=activation_fn,
            layer_args={"bias": not (layer_norm_cls == nn.LayerNorm)},
            norm_layer=[layer_norm_cls],
            norm_args=[{**layer_norm_kw, "normalized_shape": dense_units}],
        )
        self.rnn = LayerNormGRUCell(
            dense_units[0],
            recurrent_state_size,
            bias=not (layer_norm_cls == nn.LayerNorm),
            batch_first=False,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw,
        )
        self.recurrent_state_size = recurrent_state_size

    def forward(self, input: Tensor, recurrent_state: Tensor) -> Tensor:
        """
        Compute the next recurrent state from the latent state (stochastic and recurrent states) and the actions.

        Args:
            input (Tensor): the input tensor composed by the stochastic state and the actions concatenated together.
            recurrent_state (Tensor): the previous recurrent state.

        Returns:
            the computed recurrent output and recurrent state.
        """
        feat = self.mlp(input)
        out = self.rnn(feat, recurrent_state)
        return out


class BlockDiagonalRecurrentModel(nn.Module):
    """Block diagonal recurrent model for the BlockDiagonalRSSM.

    This implementation uses block diagonal linear transformations in the core computation,
    similar to the JAX implementation. It replaces the standard GRU with a more complex
    block-wise GRU-like computation that operates on divided deterministic state blocks.

    Based on the JAX BlockDiagonalRSSM implementation from DreamerV3.
    """

    def __init__(
        self,
        input_size: int,
        recurrent_state_size: int,
        hidden_size: int = 2048,
        blocks: int = 8,
        activation_fn: nn.Module = nn.GELU,
        layer_norm_cls: Callable[..., nn.Module] = nn.LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        dyn_layers: int = 1,
    ) -> None:
        """Initialize the BlockDiagonalRecurrentModel.

        Args:
            input_size (int): The size of the concatenated input (stochastic_state + action).
            recurrent_state_size (int): The size of the recurrent state (must be divisible by blocks).
            hidden_size (int): The hidden size for intermediate layers. Defaults to 2048.
            blocks (int): Number of blocks to divide the computation into. Defaults to 8.
            activation_fn (nn.Module): The activation function. Defaults to nn.GELU.
            layer_norm_cls (Callable[..., nn.Module]): The layer norm class. Defaults to LayerNorm.
            layer_norm_kw (Dict[str, Any]): The kwargs for layer norm. Defaults to {"eps": 1e-3}.
            dyn_layers (int): Number of dynamic layers. Defaults to 1.

        Raises:
            ValueError: If recurrent_state_size is not divisible by blocks.
        """
        super().__init__()

        if recurrent_state_size % blocks != 0:
            raise ValueError(f"recurrent_state_size ({recurrent_state_size}) must be divisible by blocks ({blocks})")

        self.input_size = input_size
        self.recurrent_state_size = recurrent_state_size
        self.hidden_size = hidden_size
        self.blocks = blocks
        self.activation_fn = activation_fn
        self.dyn_layers = dyn_layers

        # Two input projections: deterministic state and concatenated input (stoch+action)
        self.deter_proj = nn.Sequential(
            nn.Linear(recurrent_state_size, hidden_size, bias=not (layer_norm_cls == nn.LayerNorm)),
            layer_norm_cls(hidden_size, **layer_norm_kw),
            activation_fn(),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=not (layer_norm_cls == nn.LayerNorm)),
            layer_norm_cls(hidden_size, **layer_norm_kw),
            activation_fn(),
        )

        # Calculate the size after concatenating deter_blocks with combined_features
        # Each block will have size: (recurrent_state_size//blocks) + 2*hidden_size
        block_input_size = (recurrent_state_size // blocks) + 2 * hidden_size
        total_concat_size = blocks * block_input_size

        # First projection to map from concatenated size back to recurrent_state_size
        self.concat_proj = BlockLinear(
            total_concat_size, recurrent_state_size, blocks, bias=not (layer_norm_cls == nn.LayerNorm)
        )

        # Dynamic layers with block diagonal structure
        self.dyn_layers_list = nn.ModuleList()
        for i in range(dyn_layers):
            self.dyn_layers_list.append(
                nn.Sequential(
                    BlockLinear(
                        recurrent_state_size, recurrent_state_size, blocks, bias=not (layer_norm_cls == nn.LayerNorm),
                        ),
                    layer_norm_cls(recurrent_state_size, **layer_norm_kw),
                    activation_fn(),
                )
            )

        # Final GRU-like transformation with block diagonal structure
        self.gru_proj = BlockLinear(recurrent_state_size, 3 * recurrent_state_size, blocks, bias=True)

    def forward(self, input: Tensor, recurrent_state: Tensor) -> Tensor:
        """Compute the next recurrent state using block diagonal transformations.

        This method uses a simplified approach that processes the concatenated input as a whole
        while maintaining the block diagonal structure:
        1. Project deterministic state and concatenated input separately
        2. Combine features and repeat for each block
        3. Combine with block-grouped deterministic state
        4. Apply block diagonal dynamic layers
        5. Apply block diagonal GRU gating

        Args:
            input (Tensor): Input tensor (stochastic_state + actions concatenated).
            recurrent_state (Tensor): Previous recurrent state (deterministic state).

        Returns:
            Tensor: Updated recurrent state (deterministic state).
        """
        batch_shape = input.shape[:-1]

        # Project deterministic state and concatenated input
        x0 = self.deter_proj(recurrent_state)  # deterministic projection
        x1 = self.input_proj(input)  # concatenated input projection

        # Concatenate projections (equivalent to JAX approach but simplified)
        combined_features = torch.cat([x0, x1], dim=-1)  # [..., 2*hidden_size]

        # Add singleton dimension and repeat for blocks
        combined_features = combined_features.unsqueeze(-2)  # [..., 1, 2*hidden_size]
        combined_features = combined_features.repeat(
            *([1] * len(batch_shape)), self.blocks, 1
        )  # [..., blocks, 2*hidden_size]

        # Helper functions for block operations (matching JAX flat2group and group2flat)
        def flat2group(x):
            return x.reshape(*batch_shape, self.blocks, -1)

        def group2flat(x):
            return x.reshape(*batch_shape, -1)

        # Convert deterministic state to block format and concatenate with features
        # (matching JAX: group2flat(jnp.concatenate([flat2group(deter), x], -1)))
        deter_blocks = flat2group(recurrent_state)  # [..., blocks, deter_size//blocks]
        x = torch.cat([deter_blocks, combined_features], dim=-1)  # [..., blocks, deter_size//blocks + 2*hidden_size]
        x = group2flat(x)  # [..., blocks * (deter_size//blocks + 2*hidden_size)]

        # Project concatenated features back to recurrent_state_size
        x = self.concat_proj(x)  # [..., recurrent_state_size]
        # TODO: I would actually project each block separately instead of the whole block tensor.

        # Apply dynamic layers with block diagonal structure
        for dyn_layer in self.dyn_layers_list:
            x = dyn_layer(x)

        # Apply final GRU transformation with block diagonal structure
        gru_output = self.gru_proj(x)  # [..., 3 * recurrent_state_size]

        # Convert to block format for gate splitting (matching JAX: jnp.split(flat2group(x), 3, -1))
        gru_blocks = flat2group(gru_output)  # [..., blocks, 3 * (recurrent_state_size // blocks)]
        block_size = self.recurrent_state_size // self.blocks
        gru_blocks = gru_blocks.reshape(*batch_shape, self.blocks, 3, block_size)

        # Split gates and flatten back (matching JAX: [group2flat(x) for x in gates])
        reset_blocks, cand_blocks, update_blocks = gru_blocks.unbind(dim=-2)
        reset = group2flat(reset_blocks)
        cand = group2flat(cand_blocks)
        update = group2flat(update_blocks)

        # Apply gate functions (matching JAX implementation exactly)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)  # Note: -1 bias like in JAX

        # Update deterministic state (matching JAX: deter = update * cand + (1 - update) * deter)
        new_recurrent_state = update * cand + (1 - update) * recurrent_state

        return new_recurrent_state


class RSSM(nn.Module):
    """RSSM model for the model-base Dreamer agent.

    References:
        - [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
        - [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193)
    """

    def __init__(
        self,
        recurrent_model: RecurrentModel | _FabricModule,
        representation_model: nn.Module | _FabricModule,
        transition_model: nn.Module | _FabricModule,
        discrete: int = 32,
        unimix: float = 0.01,
        learnable_initial_recurrent_state: bool = True,
    ) -> None:
        """
        Initialize the RSSM.

        Args:
            recurrent_model (RecurrentModel | _FabricModule): The recurrent model of the RSSM model.
            representation_model (nn.Module | _FabricModule): The representation model composed by a
                multi-layer perceptron to compute the stochastic part of the latent state.
            transition_model (nn.Module | _FabricModule): The transition model composed by a
                multi-layer perceptron to predict the stochastic part of the latent state.
            discrete (int): The size of the Categorical variables. Defaults to 32.
            unimix (float): The percentage of uniform distribution to inject into the categorical
                distribution over states, i.e. given some logits `l` and probabilities `p = softmax(l)`,
                then `p = (1 - self.unimix) * p + self.unimix * unif`, where `unif = 1 / self.discrete`.
                Defaults to 0.01.
            learnable_initial_recurrent_state (bool): Whether to make the initial recurrent state learnable.
                Defaults to True.
        """
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.discrete = discrete
        self.unimix = unimix
        if learnable_initial_recurrent_state:
            self.initial_recurrent_state = nn.Parameter(
                torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "initial_recurrent_state", torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
            )

    def get_initial_states(self, batch_shape: Sequence[int] | torch.Size) -> Tuple[Tensor, Tensor]:
        """
        Get initial recurrent and stochastic states for the RSSM.

        Args:
            batch_shape (Sequence[int] | torch.Size): Shape of the batch dimensions.

        Returns:
            Tuple[Tensor, Tensor]: Initial recurrent state and initial posterior state.
        """
        initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)
        initial_posterior = self._transition(initial_recurrent_state, sample_state=False)[1]
        return initial_recurrent_state, initial_posterior

    def dynamic(
        self, posterior: Tensor, recurrent_state: Tensor, action: Tensor, embedded_obs: Tensor, is_first: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Perform one step of the dynamic learning:
            Recurrent model: compute the recurrent state from the previous latent space, the action taken by the agent,
                i.e., it computes the deterministic state (or ht).
            Transition model: predict the prior from the recurrent output.
            Representation model: compute the posterior from the recurrent state and from
                the embedded observations provided by the environment.
        For more information see [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
        and [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).

        Args:
            posterior (Tensor): the stochastic state computed by the representation model (posterior). It is expected
                to be of dimension `[stoch_size, self.discrete]`, which by default is `[32, 32]`.
            recurrent_state (Tensor): a tuple representing the recurrent state of the recurrent model.
            action (Tensor): the action taken by the agent.
            embedded_obs (Tensor): the embedded observations provided by the environment.
            is_first (Tensor): if this is the first step in the episode.

        Returns:
            The recurrent state (Tensor): the recurrent state of the recurrent model.
            The posterior stochastic state (Tensor): computed by the representation model
            The prior stochastic state (Tensor): computed by the transition model
            The logits of the posterior state (Tensor): computed by the transition model from the recurrent state.
            The logits of the prior state (Tensor): computed by the transition model from the recurrent state.
            from the recurrent state and the embbedded observation.
        """
        action = (1 - is_first) * action

        initial_recurrent_state, initial_posterior = self.get_initial_states(recurrent_state.shape[:2])
        recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)

        recurrent_state = self.recurrent_model(torch.cat((posterior, action), -1), recurrent_state)
        prior_logits, prior = self._transition(recurrent_state)
        posterior_logits, posterior = self._representation(recurrent_state, embedded_obs)
        return recurrent_state, posterior, prior, posterior_logits, prior_logits

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        """
        Apply uniform mixing to categorical distribution logits to prevent overconfident predictions.

        This method injects a small amount of uniform distribution into the categorical
        distribution to improve exploration and prevent collapse.

        Args:
            logits (Tensor): Input logits with shape [..., discrete_size] or [..., stochastic_size, discrete_size].

        Returns:
            Tensor: Mixed logits with uniform distribution injected, flattened to [..., stochastic_size * discrete_size].

        Raises:
            RuntimeError: If logits tensor doesn't have 3 or 4 dimensions.
        """
        dim = logits.dim()
        if dim == 3:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete)
        elif dim != 4:
            raise RuntimeError(f"The logits expected shape is 3 or 4: received a {dim}D tensor")
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        logits = logits.view(*logits.shape[:-2], -1)
        return logits

    def _representation(self, recurrent_state: Tensor, embedded_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_state (Tensor): the recurrent state of the recurrent model, i.e.,
                what is called h or deterministic state in
                [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
            embedded_obs (Tensor): the embedded real observations provided by the environment.

        Returns:
            logits (Tensor): the logits of the distribution of the posterior state.
            posterior (Tensor): the sampled posterior stochastic state.
        """
        logits: Tensor = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete)

    def _transition(self, recurrent_out: Tensor, sample_state=True) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_out (Tensor): the output of the recurrent model, i.e., the deterministic part of the latent space
            sampler_state (bool): whether or not to sample the stochastic state.
                Default to True

        Returns:
            logits (Tensor): the logits of the distribution of the prior state.
            prior (Tensor): the sampled prior stochastic state.
        """
        logits: Tensor = self.transition_model(recurrent_out)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete, sample=sample_state)

    def imagination(self, prior: Tensor, recurrent_state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        One-step imagination of the next latent state using the transition model.

        This method can be used iteratively to imagine trajectories in the latent space
        for planning and learning without environment interaction.

        Args:
            prior (Tensor): Prior stochastic state with shape [batch_size, stochastic_size].
            recurrent_state (Tensor): Recurrent state with shape [batch_size, recurrent_size].
            actions (Tensor): Actions to apply with shape [batch_size, action_dim].

        Returns:
            Tuple[Tensor, Tensor]: Imagined prior state and updated recurrent state.
        """
        recurrent_state = self.recurrent_model(torch.cat((prior, actions), -1), recurrent_state)
        _, imagined_prior = self._transition(recurrent_state)
        return imagined_prior, recurrent_state


class Actor(nn.Module):
    """
    DreamerV3 policy network that maps latent states to an action distribution.

    The actor consists of an MLP trunk followed by an `ActionHead` that
    parameterizes a torch.distributions.Distribution compatible with the
    provided `action_space`. Both continuous (`spaces.Box`) and discrete
    (`spaces.Discrete`, `spaces.MultiDiscrete`, `spaces.MultiBinary`) spaces are
    supported via configurable distribution heads. Categorical heads can mix in
    uniform probability mass via `unimix` to encourage exploration.
    """

    def __init__(
        self,
        action_space: spaces.Space,
        latent_state_size: int,
        dense_units: int = 1024,
        activation: nn.Module = nn.SiLU,
        mlp_layers: int = 5,
        layer_norm_cls: Callable[..., nn.Module] = nn.LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        box_head_distribution: Literal["normal", "tanh_normal", "scaled_normal", "mse"] = "scaled_normal",
        categorical_head_distribution: Literal["categorical", "one_hot_categorical"] = "one_hot_categorical",
        min_std: float = 0.1,
        max_std: float = 1.0,
        init_std: float = 2.0,
        mean_scale: float = 1.0,
        unimix: float = 0.01,
    ) -> None:
        """
        Initialize the Actor.

        Args:
            action_space (spaces.Space): Gymnasium action space defining the action
                parameterization (continuous or discrete).
            latent_state_size (int): Size of the latent input state
                (stochastic_size + recurrent_state_size).
            dense_units (int): Hidden width of the MLP trunk. Defaults to 1024.
            activation (nn.Module): Nonlinearity used in the MLP trunk. Defaults to `nn.SiLU`.
            mlp_layers (int): Number of MLP layers in the trunk. Defaults to 5.
            layer_norm_cls (Callable[..., nn.Module]): Layer normalization class applied in the trunk.
                Defaults to `LayerNorm`.
            layer_norm_kw (Dict[str, Any]): Keyword args passed to `layer_norm_cls`.
                Defaults to {"eps": 1e-3}.
            box_head_distribution (Literal["normal", "tanh_normal", "scaled_normal", "mse"]):
                Distribution type for `spaces.Box` actions. Defaults to "scaled_normal".
            categorical_head_distribution (Literal["categorical", "one_hot_categorical"]):
                Distribution type for discrete actions. Defaults to "one_hot_categorical".
            min_std (float): Minimum standard deviation (or equivalent) for continuous actions.
            max_std (float): Maximum standard deviation (or equivalent) for continuous actions.
            init_std (float): Initial standard deviation offset for continuous actions.
            mean_scale (float): The scale of the mean for the actions (only used for scaled_normal). E.g. if scale is 1,
                then the mean is in range [-1, 1], but if scale is 2, then the mean is in range [-2, 2].
            unimix (float): Uniform mixing factor added to categorical probabilities for
                discrete heads (0 disables, 0.01 by default).
        """
        super().__init__()
        self.model = MLP(
            input_dims=latent_state_size,
            output_dim=None,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=activation,
            flatten_dim=None,
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        self.mlp_head = ActionHead(
            in_features=dense_units,
            action_space=action_space,
            box_head_distribution=box_head_distribution,
            categorical_head_distribution=categorical_head_distribution,
            min_std=min_std,
            max_std=max_std,
            init_std=init_std,
            mean_scale=mean_scale,
            unimix=unimix,
        )

    def forward(self, state: Tensor) -> Distribution:
        """
        Compute the action distribution given latent state features.

        Args:
            state (Tensor): Latent state tensor of shape [..., latent_state_size].

        Returns:
            Distribution: A `torch.distributions.Distribution` over actions compatible with
            `action_space`. Sample with `.sample()` during training or use deterministic
            statistics where available (e.g., `.mean`, `.probs`, or `.mode`).
        """
        return self.mlp_head(self.model(state))


class MultiEncoder(nn.Module):
    """Multi-modal encoder that combines CNN and MLP encoders.

    This encoder can handle multiple types of inputs by using separate CNN and MLP encoders,
    and concatenates their outputs. At least one encoder must be provided.
    """

    def __init__(
        self,
        cnn_encoder: Optional[nn.Module],
        mlp_encoder: Optional[nn.Module],
    ) -> None:
        """Initialize the MultiEncoder.

        Args:
            cnn_encoder (Optional[nn.Module]): CNN encoder for image-like inputs. Must have
                `input_dim` and `output_dim` attributes if provided.
            mlp_encoder (Optional[nn.Module]): MLP encoder for vector inputs. Must have
                `input_dim` and `output_dim` attributes if provided.

        Raises:
            ValueError: If both encoders are None.
            AttributeError: If provided encoders don't have required `input_dim` or `output_dim` attributes.
        """
        super().__init__()
        if cnn_encoder is None and mlp_encoder is None:
            raise ValueError("There must be at least one encoder, both cnn and mlp encoders are None")
        self.has_cnn_encoder = False
        self.has_mlp_encoder = False
        if cnn_encoder is not None:
            if getattr(cnn_encoder, "input_dim", None) is None:
                raise AttributeError(
                    "`cnn_encoder` must contain the `input_dim` attribute representing "
                    "the dimension of the input tensor"
                )
            if getattr(cnn_encoder, "output_dim", None) is None:
                raise AttributeError(
                    "`cnn_encoder` must contain the `output_dim` attribute representing "
                    "the dimension of the output tensor"
                )
            self.has_cnn_encoder = True
        if mlp_encoder is not None:
            if getattr(mlp_encoder, "input_dim", None) is None:
                raise AttributeError(
                    "`mlp_encoder` must contain the `input_dim` attribute representing "
                    "the dimension of the input tensor"
                )
            if getattr(mlp_encoder, "output_dim", None) is None:
                raise AttributeError(
                    "`mlp_encoder` must contain the `output_dim` attribute representing "
                    "the dimension of the output tensor"
                )
            self.has_mlp_encoder = True
        self.has_both_encoders = self.has_cnn_encoder and self.has_mlp_encoder
        self.cnn_encoder = cnn_encoder
        self.mlp_encoder = mlp_encoder
        self.cnn_input_dim = self.cnn_encoder.input_dim if self.cnn_encoder is not None else None
        self.mlp_input_dim = self.mlp_encoder.input_dim if self.mlp_encoder is not None else None
        self.cnn_output_dim = self.cnn_encoder.output_dim if self.cnn_encoder is not None else 0
        self.mlp_output_dim = self.mlp_encoder.output_dim if self.mlp_encoder is not None else 0
        self.output_dim = self.cnn_output_dim + self.mlp_output_dim

    @property
    def cnn_keys(self) -> Sequence[str]:
        """Get the keys handled by the CNN encoder.

        Returns:
            Sequence[str]: List of keys that the CNN encoder processes, empty if no CNN encoder.
        """
        return self.cnn_encoder.keys if self.cnn_encoder is not None else []

    @property
    def mlp_keys(self) -> Sequence[str]:
        """Get the keys handled by the MLP encoder.

        Returns:
            Sequence[str]: List of keys that the MLP encoder processes, empty if no MLP encoder.
        """
        return self.mlp_encoder.keys if self.mlp_encoder is not None else []

    def forward(self, obs: TensorDict, *args, **kwargs) -> Tensor:
        """Forward pass through the multi-encoder.

        Processes the input observations through available encoders and concatenates their outputs.

        Args:
            obs (TensorDict): Dictionary of input observations with string keys and tensor values.
            *args: Additional positional arguments passed to the encoders.
            **kwargs: Additional keyword arguments passed to the encoders.

        Returns:
            Tensor: Concatenated output from available encoders, or output from single encoder if only one is present.
        """
        if self.has_cnn_encoder:
            cnn_out = self.cnn_encoder(obs, *args, **kwargs)
        if self.has_mlp_encoder:
            mlp_out = self.mlp_encoder(obs, *args, **kwargs)

        if self.has_both_encoders:
            return torch.cat((cnn_out, mlp_out), -1)
        elif self.has_cnn_encoder:
            return cnn_out
        else:
            return mlp_out


class MultiDecoder(nn.Module):
    """Multi-modal decoder that combines CNN and MLP decoders.

    This decoder can reconstruct multiple types of outputs by using separate CNN and MLP decoders.
    At least one decoder must be provided.
    """

    def __init__(
        self,
        cnn_decoder: Optional[nn.Module],
        mlp_decoder: Optional[nn.Module],
    ) -> None:
        """Initialize the MultiDecoder.

        Args:
            cnn_decoder (Optional[nn.Module]): CNN decoder for image-like outputs. Should have a `keys`
                attribute if provided.
            mlp_decoder (Optional[nn.Module]): MLP decoder for vector outputs. Should have a `keys`
                attribute if provided.

        Raises:
            ValueError: If both decoders are None.
        """
        super().__init__()
        if cnn_decoder is None and mlp_decoder is None:
            raise ValueError("There must be an decoder, both cnn and mlp decoders are None")
        self.cnn_decoder = cnn_decoder
        self.mlp_decoder = mlp_decoder

    @property
    def cnn_keys(self) -> Sequence[str]:
        """Get the keys handled by the CNN decoder.

        Returns:
            Sequence[str]: List of keys that the CNN decoder reconstructs, empty if no CNN decoder.
        """
        return self.cnn_decoder.keys if self.cnn_decoder is not None else []

    @property
    def mlp_keys(self) -> Sequence[str]:
        """Get the keys handled by the MLP decoder.

        Returns:
            Sequence[str]: List of keys that the MLP decoder reconstructs, empty if no MLP decoder.
        """
        return self.mlp_decoder.keys if self.mlp_decoder is not None else []

    def forward(self, x: Tensor) -> TensorDict:
        """Forward pass through the multi-decoder.

        Reconstructs observations using available decoders and combines their outputs.

        Args:
            x (Tensor): Input tensor to be decoded.

        Returns:
            TensorDict: Dictionary containing reconstructed observations from all available decoders.
        """
        reconstructed_obs = TensorDict({}, batch_size=x.shape[:-1], device=x.device)
        if self.cnn_decoder is not None:
            reconstructed_cnn_obs = self.cnn_decoder(x)
            reconstructed_obs.update(reconstructed_cnn_obs)
        if self.mlp_decoder is not None:
            reconstructed_mlp_obs = self.mlp_decoder(x)
            reconstructed_obs.update(reconstructed_mlp_obs)
        return reconstructed_obs


class Normalize(nn.Module):
    """
    PyTorch equivalent of the JAX Normalize class for adaptive normalization.

    This class provides normalization functionality that maintains running statistics
    and returns normalized values. Unlike Moments which returns offset/scale parameters,
    Normalize directly returns the mean and standard deviation (or equivalent scale parameters)
    for different normalization strategies.

    The class supports multiple normalization implementations:
    - 'none': No normalization (returns 0.0, 1.0)
    - 'meanstd': Mean-variance normalization with bias correction
    - 'perc': Percentile-based scaling using configurable quantiles

    All running statistics are stored as PyTorch buffers with bias correction support.
    """

    def __init__(
        self,
        impl: str,
        rate: float = 0.01,
        limit: float = 1e-8,
        perclo: float = 5.0,
        perchi: float = 95.0,
        debias: bool = True,
    ):
        """
        Initialize the Normalize module.

        Args:
            impl (str): Normalization implementation to use. Options:
                - 'none': No normalization
                - 'meanstd': Mean and standard deviation normalization
                - 'perc': Percentile-based scaling
            rate (float): Learning rate for exponential moving averages. Default: 0.01
            limit (float): Minimum value for scale/std to prevent division by zero. Default: 1e-8
            perclo (float): Lower percentile for percentile-based methods (0-100). Default: 5.0
            perchi (float): Upper percentile for percentile-based methods (0-100). Default: 95.0
            debias (bool): Whether to apply bias correction to statistics. Default: True

        Raises:
            NotImplementedError: If an unsupported implementation is specified.
        """
        super().__init__()
        self.impl = impl
        self.rate = rate
        self.limit = limit
        self.perclo = perclo
        self.perchi = perchi
        self.debias = debias
        self.register_buffer("device_tensor", torch.zeros((), dtype=torch.float32))

        # Register bias correction buffer if needed
        if self.debias and self.impl != "none":
            self.register_buffer("corr", torch.zeros((), dtype=torch.float32))

        # Register implementation-specific buffers
        if self.impl == "none":
            # No buffers needed
            pass
        elif self.impl == "meanstd":
            self.register_buffer("mean", torch.zeros((), dtype=torch.float32))
            self.register_buffer("sqrs", torch.zeros((), dtype=torch.float32))
        elif self.impl == "perc":
            self.register_buffer("lo", torch.zeros((), dtype=torch.float32))
            self.register_buffer("hi", torch.zeros((), dtype=torch.float32))
        else:
            raise NotImplementedError(f"Normalization implementation {self.impl} not implemented")

    def forward(self, x: Tensor, fabric: Fabric, update: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Forward pass that optionally updates statistics and returns normalization parameters.

        Args:
            x (Tensor): Input tensor to potentially update statistics with
            fabric: Fabric instance to gather the input tensor from all processes
            update (bool): Whether to update running statistics with the input data. Default: True

        Returns:
            Tuple[Tensor, Tensor]: A tuple of (mean, scale) tensors:
                - mean: Center value (mean for meanstd, lower bound for perc, 0 for none)
                - scale: Scale value (std for meanstd, span for perc, 1 for none)
        """
        if update:
            gathered_x = fabric.all_gather(x).float().detach()
            self.update(gathered_x)
        return self.stats()

    def update(self, x: Tensor) -> None:
        """
        Update running statistics with new data using exponential moving averages.

        Args:
            x (Tensor): Input data to update statistics with
        """
        if self.impl == "none":
            pass
        elif self.impl == "meanstd":
            self._update_buffer(self.mean, torch.mean(x))
            self._update_buffer(self.sqrs, torch.mean(x * x))
        elif self.impl == "perc":
            self._update_buffer(self.lo, self._perc(x, self.perclo))
            self._update_buffer(self.hi, self._perc(x, self.perchi))
        else:
            raise NotImplementedError(self.impl)

        # Update bias correction if enabled
        if self.debias and self.impl != "none":
            self._update_buffer(self.corr, 1.0)

    def stats(self) -> Tuple[Tensor, Tensor]:
        """
        Return current normalization statistics as mean and scale parameters.

        Returns:
            Tuple[Tensor, Tensor]: A tuple of (mean, scale) where:
                - mean (Tensor): Center parameter
                - scale (Tensor): Scale parameter
        """
        # Compute bias correction factor
        corr = 1.0
        if self.debias and self.impl != "none":
            corr = 1.0 / torch.maximum(torch.tensor(self.rate, device=self.device_tensor.device), self.corr)

        if self.impl == "none":
            return (
                torch.tensor(0.0, device=self.device_tensor.device),
                torch.tensor(1.0, device=self.device_tensor.device),
            )
        elif self.impl == "meanstd":
            mean = self.mean * corr
            std = torch.sqrt(torch.relu(self.sqrs * corr - mean**2))
            std = torch.maximum(torch.tensor(self.limit, device=self.device_tensor.device), std)
            return mean.detach(), std.detach()
        elif self.impl == "perc":
            lo = self.lo * corr
            hi = self.hi * corr
            span = hi - lo
            span = torch.maximum(torch.tensor(self.limit, device=self.device_tensor.device), span)
            return lo.detach(), span.detach()
        else:
            raise NotImplementedError(self.impl)

    def _perc(self, x: Tensor, q: float) -> Tensor:
        """
        Compute percentile of input tensor.

        Args:
            x (Tensor): Input tensor
            q (float): Percentile to compute (0-100)

        Returns:
            Tensor: Percentile value
        """
        return torch.quantile(x, q / 100.0)

    def _update_buffer(self, buffer: Tensor, value) -> None:
        """
        Update a buffer using exponential moving average.

        Args:
            buffer (Tensor): Buffer to update
            value: New value to incorporate (Tensor, int, or float)
        """
        if isinstance(value, (int, float)):
            value = torch.tensor(value, device=buffer.device, dtype=buffer.dtype)
        buffer.mul_(1 - self.rate).add_(value.detach(), alpha=self.rate)


class BlockLinear(nn.Linear):
    """Block diagonal linear layer for efficient block-wise transformations.

    This layer divides the input into `blocks` groups and applies separate linear
    transformations to each block independently, then concatenates the results.
    This is equivalent to a block diagonal matrix multiplication but more efficient.

    Based on the JAX implementation in DreamerV3's BlockDiagonalRSSM.
    """

    def __init__(
        self, in_features: int, out_features: int, blocks: int, bias: bool = True, device=None, dtype=None, **kwargs
    ) -> None:
        """Initialize the BlockLinear layer.

        Args:
            in_features (int): Total input dimension (must be divisible by blocks).
            out_features (int): Total output dimension (must be divisible by blocks).
            blocks (int): Number of blocks to divide the computation into.
            bias (bool): Whether to include bias terms. Defaults to True.
            device: Device to place parameters on.
            dtype: Data type for parameters.
            **kwargs: Additional arguments (for compatibility).

        Raises:
            ValueError: If in_features or out_features is not divisible by blocks.
        """

        if in_features % blocks != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by blocks ({blocks})")
        if out_features % blocks != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by blocks ({blocks})")

        # Store block-specific attributes
        self.blocks = blocks
        self.input_block_size = in_features // blocks
        self.output_block_size = out_features // blocks

        # Set a flag to prevent reset_parameters from running during parent init
        self._block_linear_init = False

        # Initialize parent nn.Linear with dummy parameters (we'll override them)
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)

        # Replace the standard weight with our block-structured weight
        # Shape: [blocks, input_block_size, output_block_size] to match JAX einsum pattern
        del self.weight  # Remove the standard linear weight
        self.weight = nn.Parameter(
            torch.empty(self.blocks, self.input_block_size, self.output_block_size, device=device, dtype=dtype)
        )

        # Create bias if needed (reuse parent's bias parameter)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # Now we can safely initialize parameters
        self._block_linear_init = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the parameters using standard linear layer initialization."""
        # Only run our custom initialization if we've finished setting up the block structure
        if not getattr(self, "_block_linear_init", False):
            return

        # Initialize each block's weights independently like separate Linear layers
        for i in range(self.blocks):
            nn.init.kaiming_uniform_(self.weight[i], a=5.0**0.5)

        if self.bias is not None:
            # Initialize bias for each block independently
            fan_in = self.input_block_size
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            for i in range(self.blocks):
                start_idx = i * self.output_block_size
                end_idx = (i + 1) * self.output_block_size
                nn.init.uniform_(self.bias[start_idx:end_idx], -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through block diagonal linear transformation.

        Args:
            input (Tensor): Input tensor with shape [..., in_features].

        Returns:
            Tensor: Output tensor with shape [..., out_features].

        Raises:
            ValueError: If input's last dimension doesn't match in_features.
        """
        if input.shape[-1] != self.in_features:
            raise ValueError(f"Input last dimension ({input.shape[-1]}) doesn't match in_features ({self.in_features})")

        # Split input into blocks: [..., in_features] -> [..., blocks, input_block_size]
        batch_shape = input.shape[:-1]
        x_blocks = input.reshape(*batch_shape, self.blocks, self.input_block_size)

        # Vectorized matrix multiplication using JAX's einsum pattern: '...ki,kio->...ko'
        # x_blocks: [..., blocks, input_block_size], self.weight: [blocks, input_block_size, output_block_size]
        output = torch.einsum("...ki,kio->...ko", x_blocks, self.weight)  # [..., blocks, output_block_size]

        # Reshape to final output shape before adding bias (matching JAX)
        output = output.reshape(*batch_shape, self.out_features)  # [..., out_features]

        # Add bias if present (pre-computed tensor, no runtime concatenation)
        if self.bias is not None:
            output = output + self.bias

        return output
