"""
Collection of utility functions.
"""

import functools
import os
from typing import Any, Dict, List, Optional, Type, Union

import torch
from gymnasium import Env, Wrapper
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv, VectorWrapper
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from tensordict import TensorDict
from wandb.integration.lightning.fabric import WandbLogger


def merge_dicts(x: Union[Dict, DictConfig], y: Union[Dict, DictConfig]) -> Dict:
    """
    Merge two dictionaries, updating the first dictionary `x`with the values of the second dictionary `y`.

    If a query is present in both dicts and the values are lists, then they are concatenated.
    Note: This is only done on the first level of the dictionary.

    Args:
        x (Union[Dict, DictConfig]): The first dictionary to be updated.
        y (Union[Dict, DictConfig]): The dictionary containing values to update the first dictionary.

    Returns:
        Dict: The updated dictionary containing values from both dictionaries.

    Example:
        >>> x = {'a': 1, 'b': [2, 3]}
        >>> y = {'b': [4, 5], 'c': 6}
        >>> updated_dict = merge_dicts(x, y)
        >>> print(updated_dict)
        {'a': 1, 'b': [2, 3, 4, 5], 'c': 6}
    """
    x_new = OmegaConf.to_container(x.copy()) if isinstance(x, DictConfig) else x.copy()
    for key, value in y.items():
        if key in x_new:
            if isinstance(x_new[key], (list, ListConfig)) and isinstance(value, (list, ListConfig)):
                x_new[key].extend(value)
            else:
                x_new[key] = value
        else:
            x_new[key] = value
    return x_new


def set_cuda_backend() -> None:
    """Set the CUDA backend for PyTorch to use TensorFloat32 and TF32."""
    print("Setting CUDA backend to TensorFloat32 and TF32")
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn
    torch.set_float32_matmul_precision = "medium"  # Set matmul precision to medium


def configure_wandb_logging(experiment_config: Dict[str, Any], **extra_logger_kwargs) -> WandbLogger:
    """Configure Weights & Biases logging for an experiment.

    Update the experiment logger configuration with additional logger arguments.
    Add id to the wandb name if provided.

    Args:
        experiment_config (Dict[str, Any]): Experiment configuration.
        extra_logger_kwargs (Dict[str, Any]): Additional logger arguments.

    Returns:
        WandbLogger: Weights & Biases logger.
    """
    logger_conf = experiment_config["logger"]
    experiment_id = extra_logger_kwargs.get("id", None)
    logger_conf["name"] += f"_{experiment_id}" if experiment_id else ""  # name_id
    logger_conf = merge_dicts(logger_conf, extra_logger_kwargs)
    return WandbLogger(**logger_conf, config=dict(experiment_config))


def rename_key(dictionary: Dict, old_key: str, new_key: str) -> Dict:
    """Rename a key in a dictionary.

    Args:
        dictionary (Dict): The dictionary to update.
        old_key (str): The key to rename.
        new_key (str): The new key name.

    Returns:
        Dict: The updated dictionary with the key renamed.
    """
    dictionary[new_key] = dictionary.pop(old_key)
    return dictionary


def requires_grad(model: torch.nn.Module, requires_grad: bool = True) -> None:
    """Set the requires_grad attribute of all model parameters to the given value.

    Args:
        model (torch.nn.Module): The model to update.
        requires_grad (bool, optional): The value to set requires_grad to. Defaults to True.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def display_banner() -> None:
    """Display the Imitation Learning From Observation banner."""
    banner = r"""
================ Imitation Learning From Observation ================

         .___        .__  __          __  .__
         |   | _____ |__|/  |______ _/  |_|__| ____   ____
         |   |/     \|  \   __\__  \\   __\  |/  _ \ /    \
         |   |  Y Y  \  ||  |  / __ \|  | |  (  <_> )   |  \
         |___|__|_|  /__||__| (____  /__| |__|\____/|___|  /
                   \/              \/                    \/

======================================================================

"""
    print(banner)


class Conditional:
    """
    A wrapper to conditionally execute a context manager.
    """

    def __init__(self, original_context_manager: object, condition: bool):
        """
        Initializes the ConditionalContextManager.

        Args:
            original_context_manager object: The context manager to conditionally execute.
            condition (bool): Whether to use the context manager.
        """
        self.original_context_manager = original_context_manager
        self.condition = condition

    def __enter__(self) -> Optional[object]:
        """
        Enters the wrapped context manager if the condition is `True`.

        Returns:
            Optional[object]: The result of the context manager's `__enter__`
            method, or `None` if the context manager is not executed.
        """
        if self.condition:
            return self.original_context_manager.__enter__()

    def __exit__(self, *exc_info) -> Optional[bool]:
        """
        Exits the wrapped context manager if the condition is `True`.

        Args:
            exc_info: The exception information.

        Returns:
            Optional[bool]: Returns the result of the context manager's `__exit__` or `None`
        """
        if self.condition:
            return self.original_context_manager.__exit__(*exc_info)


def restore_torch_state(func):
    """
    A decorator to save and restore the gradient state and training mode of a PyTorch model.

    This decorator ensures that the gradient computation state (enabled or disabled) and the
    training mode (training or evaluation) of the model are preserved before and after the
    execution of the wrapped function.

    Args:
        func (callable): The function to be wrapped by the decorator.

    Returns:
        callable: The wrapped function with gradient state and training mode preservation.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert "model" in kwargs, "The model argument must be provided."
        # Save the current gradient state and mode of the model
        grad_enabled = torch.is_grad_enabled()
        training_mode = kwargs["model"].training

        try:
            result = func(*args, **kwargs)
        finally:
            # Restore the gradient state and mode of the model
            torch.set_grad_enabled(grad_enabled)
            if training_mode:
                kwargs["model"].train()
            else:
                kwargs["model"].eval()

        return result

    return wrapper


def tensordict_collate(x: Union[List[TensorDict], TensorDict]) -> TensorDict:
    """Collate TensorDict.

    Args:
        x (Union[List[TensorDict], TensorDict]): List of TensorDicts or a single TensorDict.

    Returns:
        TensorDict: Stacked TensorDict.
    """
    if isinstance(x, TensorDict):
        return x
    else:
        return torch.stack(x)


def get_latest_checkpoint(checkpoint: Optional[str] = None, extension: Optional[str] = "ckpt") -> Optional[str]:
    """
    Returns the latest checkpoint.

    If `checkpoint` is a directory, the latest checkpoint is returned. If it is a file, it is returned as is.

    Args:
        checkpoint (Optional[str]): The path to the checkpoint.
        extension (Optional[str]): The extension of the checkpoint.

    Returns:
        Optional[str]: The path to the latest checkpoint file, or None if no checkpoints are found.
    """
    has_extension = lambda x: extension is None or x.endswith(extension)
    if not checkpoint:
        return None

    if os.path.isfile(checkpoint) and has_extension(checkpoint):
        return checkpoint

    if os.path.isdir(checkpoint):
        items = sorted(os.listdir(checkpoint))
        items = [f for f in items if os.path.isfile(os.path.join(checkpoint, f)) and has_extension(f)]
        return os.path.join(checkpoint, items[-1]) if items else None

    return None


def add_legacy_features_to_vector_wrapper() -> None:
    """Add legacy features of Wrapper to VectorWrapper and base vector environment classes.

    This function adds the following methods to VectorWrapper, AsyncVectorEnv, and SyncVectorEnv:
        - has_wrapper_attr
        - get_wrapper_attr
        - set_wrapper_attr

    This allows these methods to work with both wrapped and unwrapped vector environments.
    """
    # Add to VectorWrapper (for wrapped environments)
    VectorWrapper.has_wrapper_attr = Wrapper.has_wrapper_attr
    VectorWrapper.get_wrapper_attr = Wrapper.get_wrapper_attr
    VectorWrapper.set_wrapper_attr = Wrapper.set_wrapper_attr

    # Add to AsyncVectorEnv (for unwrapped async vector environments)
    AsyncVectorEnv.has_wrapper_attr = Wrapper.has_wrapper_attr
    AsyncVectorEnv.get_wrapper_attr = Wrapper.get_wrapper_attr
    AsyncVectorEnv.set_wrapper_attr = Wrapper.set_wrapper_attr

    # Add to SyncVectorEnv (for unwrapped sync vector environments)
    SyncVectorEnv.has_wrapper_attr = Wrapper.has_wrapper_attr
    SyncVectorEnv.get_wrapper_attr = Wrapper.get_wrapper_attr
    SyncVectorEnv.set_wrapper_attr = Wrapper.set_wrapper_attr


def is_wrapped(env: Union[Env, VectorEnv], wrapper_class: Type[Union[Wrapper, VectorWrapper]]) -> bool:
    """
    Check if env is wrapped with an instance of wrapper_class.

    Args:
        env (Union[Env, VectorEnv]): The environment to check.
        wrapper_class (Type[Union[Wrapper, VectorWrapper]]): The wrapper class to check for.

    Returns:
        bool: True if env is wrapped with an instance of wrapper_class, False otherwise.
    """
    while isinstance(env, (Wrapper, VectorWrapper)):
        if isinstance(env, wrapper_class):
            return True
        env = env.env  # Unwrap one level
    return False
