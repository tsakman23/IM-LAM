"""
Action descretization, transformation and probing utilities.
Some are from the Video-Pre-Training repository: https://github.com/openai/Video-Pre-Training"""

import itertools
from collections import OrderedDict
from typing import Dict, Union

import attr
import numpy as np
import torch
from torch import Tensor
from gymnasium import spaces
from tensordict import TensorDict
from tensordict.nn.distributions import TruncatedNormal
from torch.distributions import Categorical, Distribution, Independent, Normal

from ifo.common.distributions import DictDistribution, MSEDistribution, MultiCategorical


class QuantizationScheme:
    """Quantization schemes for `CameraQuantizer`."""

    LINEAR = "linear"
    MU_LAW = "mu_law"


@attr.s(auto_attribs=True)
class CameraQuantizer:
    """
    A camera quantizer that discretizes and undiscretizes a continuous camera input with y (pitch) and x (yaw)
    components.

    Attributes:
        camera_maxval (int): The maximum value of the camera action.
        camera_binsize (int): The size of the bins used for quantization. In case of mu-law quantization, it corresponds
            to the average binsize.
        quantization_scheme (str): The quantization scheme to use. Currently, two quantization schemes are supported:
            - Linear quantization (default): Camera actions are split uniformly into discrete bins.
            - Mu-law quantization: Transforms the camera action using mu-law encoding
              (https://en.wikipedia.org/wiki/%CE%9C-law_algorithm) followed by the same quantization scheme used by the
              linear scheme.
        mu (float): Mu is the parameter that defines the curvature of the mu-law encoding. Higher values of
            mu will result in a sharper transition near zero. Below are some reference values listed
            for choosing mu given a constant maxval and a desired max_precision value.

    Examples:
        maxval = 10 | max_precision = 0.5  | μ ≈ 2.93826
        maxval = 10 | max_precision = 0.4  | μ ≈ 4.80939
        maxval = 10 | max_precision = 0.25 | μ ≈ 11.4887
        maxval = 20 | max_precision = 0.5  | μ ≈ 2.7
        maxval = 20 | max_precision = 0.4  | μ ≈ 4.39768
        maxval = 20 | max_precision = 0.25 | μ ≈ 10.3194
        maxval = 40 | max_precision = 0.5  | μ ≈ 2.60780
        maxval = 40 | max_precision = 0.4  | μ ≈ 4.21554
        maxval = 40 | max_precision = 0.25 | μ ≈ 9.81152
    """

    camera_maxval: int
    camera_binsize: int
    quantization_scheme: str = attr.ib(
        default=QuantizationScheme.LINEAR,
        validator=attr.validators.in_([QuantizationScheme.LINEAR, QuantizationScheme.MU_LAW]),
    )
    mu: float = attr.ib(default=5)

    def discretize(self, xy: np.ndarray) -> np.ndarray:
        """Discretize the camera action.

        Args:
            xy (np.ndarray): The continuous camera action.

        Returns:
            np.ndarray: The discretized camera action.
        """
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_encode = np.sign(xy) * (np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu))
            v_encode *= self.camera_maxval
            xy = v_encode

        # Quantize using linear scheme
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(np.int64)

    def undiscretize(self, xy: np.ndarray) -> np.ndarray:
        """Undiscretize the camera action.

        Args:
            xy (np.ndarray): The discretized camera action.

        Returns:
            np.ndarray: The continuous camera action.
        """
        xy = xy * self.camera_binsize - self.camera_maxval

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_decode = np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            v_decode *= self.camera_maxval
            xy = v_decode
        return xy


class DiscretizeMineRLAction:
    """
    Class to transform the original MineRL action space by converting from a MultiDiscrete action space, where each
    button can be pressed independently, to a Discrete action space where only one button of each group can be pressed.
    Furthermore the camera action is discretized using a CameraQuantizer.

    This removes actions, which make no sense in the game, like pressing forward and back at the same time, which
    would result in no movement. Furthermore the camera action is discretized and clipped to make it easier to learn.
    """

    # Buttons in the original MineRL action space.
    minerl_buttons = [
        "ESC",
        "back",
        "drop",
        "forward",
        "hotbar.1",
        "hotbar.2",
        "hotbar.3",
        "hotbar.4",
        "hotbar.5",
        "hotbar.6",
        "hotbar.7",
        "hotbar.8",
        "hotbar.9",
        "inventory",
        "jump",
        "left",
        "right",
        "sneak",
        "sprint",
        "swapHands",
        "attack",
        "use",
        "pickItem",
        "camera",
    ]

    # Group of buttons of the new action space, where only one button of the button group can be pressed at a time.
    exclusive_button_groups = OrderedDict(
        hotbar=[None] + [f"hotbar.{i}" for i in range(1, 10)],
        fore_back=[None, "forward", "back"],
        left_right=[None, "left", "right"],
        sprint_sneak=[None, "sprint", "sneak"],
        use=[None, "use"],
        drop=[None, "drop"],
        attack=[None, "attack"],
        jump=[None, "jump"],
    )

    # Buttons that can be pressed, but are mutually exclusive with all other buttons and camera.
    exclusive_buttons = ["inventory", "ESC"]

    # Map the value of a exclusive button group to the index in the group.
    group_value_to_index = {
        group: {value: idx for idx, value in enumerate(values)} for group, values in exclusive_button_groups.items()
    }
    # Map the index in the group to the value of an exclusive button group.
    group_index_to_value = {
        group: {idx: value for idx, value in enumerate(values)} for group, values in exclusive_button_groups.items()
    }
    # Size of each group.
    group_sizes = {group: len(values) for group, values in exclusive_button_groups.items()}

    def __init__(
        self,
        camera_maxval: int = 10,
        camera_binsize: int = 2,
        camera_quantization_scheme: str = QuantizationScheme.MU_LAW,
        camera_mu: int = 10,
        n_camera_bins: int = 11,
    ) -> None:
        """
        Initializes DiscreteMineRLActionMapping wrapper.

        Args:
            camera_maxval (int, optional): The maximum value for the camera quantizer.
            camera_binsize (int, optional): The bin size for the camera quantizer.
            camera_quantization_scheme (str, optional): The quantization scheme for the camera quantizer.
            camera_mu (float, optional): The mu parameter for the camera quantizer.
            n_camera_bins (int, optional): The number of camera bins in the factored space.
        """
        assert n_camera_bins % 2 == 1, "The number of camera bins must be odd to ensure a null bin."
        self.n_camera_bins = n_camera_bins
        self.camera_null_bin = n_camera_bins // 2
        self.quantizer = CameraQuantizer(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            quantization_scheme=camera_quantization_scheme,
            mu=camera_mu,
        )
        self.precompute_reverse_action()
        self.camera_to_index = np.arange(n_camera_bins**2).reshape(n_camera_bins, n_camera_bins)
        self.index_to_camera = np.array(np.unravel_index(np.arange(n_camera_bins**2), (n_camera_bins, n_camera_bins))).T
        np.column_stack(np.unravel_index(np.arange(n_camera_bins**2), (n_camera_bins, n_camera_bins)))
        self.single_action_space = spaces.Dict(
            {
                "camera": spaces.Discrete(n_camera_bins**2),
                "buttons": spaces.Discrete(len(self.index_to_button_combination)),
            }
        )

    def precompute_reverse_action(self) -> None:
        """Compute the mapping from index to button combination and save in `index_to_button_combination`."""
        self.index_to_button_combination = []

        # Create permutations for all exclusive button groups and create mapping from index to the button combination.
        combinations = list(itertools.product(*reversed(self.exclusive_button_groups.values())))
        self.exclusive_button_group_combinations = len(combinations)
        for combination in combinations:
            action = np.zeros(len(self.minerl_buttons[:-1]), dtype=int)  # Exclude camera.
            for button in combination:
                if button:
                    action[self.minerl_buttons.index(button)] = 1
            self.index_to_button_combination.append(action)
        # Create mapping for all exclusive buttons.

        for exclusive_button in self.exclusive_buttons:
            action = np.zeros(len(self.minerl_buttons[:-1]), dtype=int)  # Exclude camera.
            action[self.minerl_buttons.index(exclusive_button)] = 1
            self.index_to_button_combination.append(action)

    def forward_camera(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform the camera action.

        Args:
            action (Dict[str, np.ndarray]): MineRL action.

        Returns:
            np.ndarray: Discretized camera action.
        """
        discrete_camera = self.quantizer.discretize(action["camera"])
        set_null_bin = np.where(sum([action[key] for key in self.exclusive_buttons]) > 0)
        discrete_camera[set_null_bin] = self.camera_null_bin
        return self.camera_to_index[discrete_camera[:, 0], discrete_camera[:, 1]]

    def forward_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform the button action.

        Args:
            action (Dict[str, np.ndarray]): MineRL action.

        Returns:
            np.ndarray: Transformed button action.
        """
        buttons = self.button_combination_to_index(action)
        for i, key in enumerate(self.exclusive_buttons):
            mask = action[key] > 0
            buttons[mask] = self.exclusive_button_group_combinations + i
        return buttons

    def button_combination_to_index(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert a combination of buttons to an index.

        Args:
            action (Dict[str, np.ndarray]): MineRL action.

        Returns:
            np.ndarray: Index of the button combination.
        """
        index = np.zeros_like(action["forward"], dtype=int)
        multiplier = 1
        # Calculate the index for all exclusive button groups.
        for group in self.exclusive_button_groups.keys():
            index += self.get_index(action, group) * multiplier
            multiplier *= self.group_sizes[group]
        # Calculate the index for all exclusive buttons.
        # We just add them at the end.
        for i, key in enumerate(self.exclusive_buttons):
            mask = action[key] > 0
            index[mask] = self.exclusive_button_group_combinations + i
        return index

    def get_index(self, action: Dict[str, np.ndarray], group: str) -> np.ndarray:
        """
        Get the index for a certain group of buttons.

        This is actually super trivial, but we add some minor tweaks:
            - If both buttons in a movement group are pressed, we choose none,
              because the agent can't move in two directions at the same time.
            - If multiple buttons are pressed in a group, we choose the last one.

        Args:
            action (Dict[str, np.ndarray]): MineRL action.
            group (str): Button group of `exclusive_button_groups`.

        Returns:
            np.ndarray: Index of button pressed in the group.
        """
        index = np.zeros_like(action["forward"], dtype=int)
        if group == "fore_back":
            mask = (action["forward"] + action["back"]) > 1  # If both are pressed, choose none.
            action["forward"][mask] = action["back"][mask] = 0
        elif group == "left_right":
            mask = (action["left"] + action["right"]) > 1  # If both are pressed, choose none.
            action["left"][mask] = action["right"][mask] = 0

        # If multiple buttons are pressed in a group, choose the last one. The one with the highest index.
        for choice in self.exclusive_button_groups[group]:
            if choice:
                index = np.maximum(index, action[choice] * self.group_value_to_index[group][choice])
        return index

    def forward(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward transform the MineRL action to the new action space.

        Args:
            action (Dict[str, np.ndarray]): MineRL action.

        Returns:
            Dict[str, np.ndarray]: Transformed action.
        """
        return dict(camera=self.forward_camera(action), buttons=self.forward_action(action))

    def reverse_camera(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Reverse transform the camera action.

        Args:
            action (Dict[str, np.ndarray]): Transformed action.

        Returns:
            np.ndarray: MineRL camera action.
        """
        return self.quantizer.undiscretize(self.index_to_camera[action["camera"]])

    def reverse_action(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Reverse transform the button action.

        Args:
            action (Dict[str, np.ndarray]): Transformed action.

        Returns:
            Dict[str, np.ndarray]: MineRL button action.
        """
        buttons = action["buttons"]
        # Create an action dictionary with the final dimensions for all buttons.
        minerl_action = {key: np.zeros_like(buttons, dtype=int) for key in self.minerl_buttons[:-1]}

        # Need to go through each batch item and retrieve from a precomputed dictionary
        converted_action = [self.index_to_button_combination[buttons[i]] for i in range(buttons.size)]
        converted_action = np.stack(converted_action, axis=0)
        converted_action = np.split(converted_action, converted_action.shape[-1], axis=-1)

        # Set the converted action values in the final MineRL action dictionary.
        for i, key in enumerate(self.minerl_buttons[:-1]):
            minerl_action[key] = converted_action[i].squeeze()

        return minerl_action

    def reverse(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Reverse transform the action to the original MineRL action space.

        Args:
            action (Dict[str, np.ndarray]): Transformed action.

        Returns:
            Dict[str, np.ndarray]: MineRL action.
        """
        return dict(camera=self.reverse_camera(action), **self.reverse_action(action))


def get_action_metrics(
    action_dist: Distribution,
    target_action: Union[Tensor, TensorDict],
    prefix: str = "",
    postfix: str = "",
    metrics: Dict[str, Tensor] | TensorDict = {},
) -> Dict[str, Tensor] | TensorDict:
    """Get the metrics for the action distribution.
    Calculates continuous and discrete metrics for the action distribution.
    Args:
        action_dist (Distribution): The action distribution output by the decoder network.
        target_action (Union[Tensor, TensorDict]): The ground truth action to compare against.
        prefix (str, optional): String prefix to prepend to metric names. Defaults to "".
        postfix (str, optional): String suffix to append to metric names. Defaults to "".
        metrics (Dict[str, Tensor] | TensorDict, optional): Dictionary to store computed metrics. Defaults to {}.
    Returns:
        Dict[str, Tensor] | TensorDict: Dictionary containing computed metrics for the action distribution,
            including MSE for continuous actions and accuracy for discrete actions.
    """
    if isinstance(action_dist, Independent):
        return get_action_metrics(action_dist.base_dist, target_action, prefix, postfix, metrics)
    if isinstance(action_dist, (TruncatedNormal, Normal, MSEDistribution)):
        assert isinstance(target_action, Tensor)
        metrics[f"{prefix}mse{postfix}"] = torch.nn.functional.mse_loss(action_dist.mode, target_action)
        return metrics
    elif isinstance(action_dist, Categorical):
        metrics[f"{prefix}accuracy{postfix}"] = (action_dist.mode == target_action).float().mean()
        return metrics
    elif isinstance(action_dist, MultiCategorical):
        accuracies = [
            (dist.mode == action).float().mean() for dist, action in zip(action_dist.distributions, target_action)
        ]
        metrics[f"{prefix}accuracy{postfix}"] = torch.stack(accuracies).mean()
        return metrics
    elif isinstance(action_dist, DictDistribution):
        assert isinstance(target_action, TensorDict)
        for key in target_action.keys():
            get_action_metrics(
                action_dist.distributions[key], target_action[key], prefix, postfix=f"{postfix}_{key}", metrics=metrics
            )
        return metrics
    else:
        raise NotImplementedError
