from typing import Callable, Optional

from ifo.common.utils.functions import uncenter


def atari() -> Callable:
    """Return the inverse transformation to reconstruct observations.

    Returns:
        Callable: Inverse transformation.
    """

    def transform(x):
        return uncenter(x)

    return transform


def procgen() -> Callable:
    """Return the inverse transformation to reconstruct observations.

    Returns:
        Callable: Inverse transformation.
    """

    def transform(x):
        return uncenter(x)

    return transform


def minigrid() -> Callable:
    """Return the inverse transformation to reconstruct observations.

    Returns:
        Callable: Inverse transformation.
    """

    def transform(x):
        return uncenter(x)

    return transform


def vizdoom() -> Callable:
    """Return the inverse transformation to reconstruct observations.

    Returns:
        Callable: Inverse transformation.
    """

    def transform(x):
        return uncenter(x)

    return transform


def minerl() -> Callable:
    """Return the inverse transformation to reconstruct observations.

    Returns:
        Callable: Inverse transformation.
    """

    def transform(x):
        return uncenter(x)

    return transform


def dmc() -> Callable:
    """Return the inverse transformation to reconstruct observations.

    Returns:
        Callable: Inverse transformation.
    """

    def transform(x):
        return uncenter(x)

    return transform


def dmc_distracting() -> Callable:
    """Return the inverse transformation to reconstruct observations.

    Returns:
        Callable: Inverse transformation.
    """
    return dmc()


def dcs_masked() -> Callable:
    """Return the inverse transformation to reconstruct observations.

    Returns:
        Callable: Inverse transformation.
    """

    def transform(x):
        return uncenter(x)

    return transform


def metaworld() -> Callable:
    """Return the inverse transformation to reconstruct observations.

    Returns:
        Callable: Inverse transformation.
    """

    def transform(x):
        return uncenter(x)

    return transform

def get_debug_transform(env_dataset_name: Optional[str]) -> Callable:
    """Get inverse transformation for the specified dataset or environment.

    Args:
        env_dataset_name (str): Name of the dataset or environment.

    Returns:
        Callable: Inverse transformation.
    """
    if env_dataset_name is None:
        return None
    if "ALE" in env_dataset_name:
        return atari()
    if "procgen" in env_dataset_name:
        return procgen()
    elif "MiniGrid" in env_dataset_name:
        return minigrid()
    elif "Vizdoom" in env_dataset_name:
        return vizdoom()
    elif "MineRL" in env_dataset_name:
        return minerl()
    elif "dm_control" in env_dataset_name:
        if "masked" in env_dataset_name:
            return dcs_masked()
        elif "distracting" in env_dataset_name:
            return dmc_distracting()
        else:
            return dmc()
    elif "Meta-World" in env_dataset_name:
        return metaworld()
    else:
        raise NotImplementedError("Dataset/Environment is currently not supported.")
