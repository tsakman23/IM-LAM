from typing import Dict, Literal, Tuple

import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor

ActionSpaceType = Literal["continuous", "discrete"]


def make_action_space(action_dim: int, action_space_type: ActionSpaceType) -> spaces.Space:
    """Construct the configured Gymnasium action space."""
    if action_space_type == "continuous":
        return spaces.Box(low=-1, high=1, shape=(action_dim,))
    if action_space_type == "discrete":
        return spaces.Discrete(action_dim)
    raise ValueError(f"Unsupported action space type: {action_space_type}")


def prepare_action_targets(actions: Tensor, action_space_type: ActionSpaceType) -> Tensor:
    """Normalize action labels for the configured distribution."""
    if action_space_type == "discrete":
        return actions.long().reshape(-1)
    return actions


def action_probe_loss_and_metrics(
    predictions: Tensor,
    actions: Tensor,
    action_space_type: ActionSpaceType,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute an action probe objective and detached diagnostics."""
    if action_space_type == "discrete":
        targets = prepare_action_targets(actions, action_space_type)
        loss = F.cross_entropy(predictions, targets)
        accuracy = (predictions.argmax(dim=-1) == targets).float().mean()
        return loss, {
            "action_decoder_ce": loss.detach(),
            "action_decoder_accuracy": accuracy.detach(),
        }

    loss = F.mse_loss(predictions, actions)
    return loss, {"action_decoder_mse": loss.detach()}
