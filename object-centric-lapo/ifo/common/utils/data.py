import os
from typing import Optional, Union

from torch.utils.data import Dataset

from ifo.common.data import (
    DeepMindControlSuiteDataset,
    MaskedDistractingControlSuiteDataset,
    MaskedFroggerDataset,
    MaskedFrostbiteDataset,
    MetaWorldDataset,
    ProcgenDataset,
)
from ifo.common.transforms import get_dataset_transform
from ifo.common.transforms.data import DictTransform


def get_procgen_dataset(
    env_name: str,
    split: str,
    cache_dir: Optional[str] = None,
    block_size: Optional[int] = None,
    transform: Optional[DictTransform] = None,
) -> ProcgenDataset:
    """Get the available Procgen dataset on Hugging Face.

    Args:
        env_name (str): The name of the environment.
        split (str): The dataset split, e.g., 'train' or 'test'.
        cache_dir (Optional[str]): Directory to cache the dataset.
        block_size (Optional[int]): The block size for the dataset.
        transform (Optional[DictTransform]): The transformation to apply to the dataset.

    Returns:
        ProcgenDataset: Available Procgen datasets on Hugging Face.
    """

    datasets = [
        "bigfish",
        "bossfight",
        "caveflyer",
        "chaser",
        "climber",
        "coinrun",
        "dodgeball",
        "fruitbot",
        "heist",
        "jumper",
        "leaper",
        "maze",
        "miner",
        "ninja",
        "plunder",
        "starpilot",
    ]
    name = env_name.split("-")[1]
    assert name in datasets, f"Environment {name} not found in the Procgen dataset. Available environments: {datasets}"
    return ProcgenDataset(
        name=name,
        split=split,
        cache_dir=cache_dir,
        num_proc=8,
        block_size=block_size,
        columns=["observation", "action"],
        transform=transform,
    )


def get_dmc_dataset(
    env_name: str,
    split: str,
    cache_dir: Optional[str] = None,
    block_size: Optional[int] = None,
    transform: Optional[DictTransform] = None,
) -> DeepMindControlSuiteDataset:
    """Get the available DeepMind Control Suite dataset.

    Args:
        env_name (str): The name of the environment.
        split (str): The dataset split, e.g., 'train' or 'test'.
        cache_dir (Optional[str]): Directory to cache the dataset.
        block_size (Optional[int]): The block size for the dataset.
        transform (Optional[DictTransform]): The transformation to apply to the dataset.

    Returns:
        Dataset: Available DeepMind Control Suite dataset.
    """

    datasets = [
        "acrobot_swingup",
        "cartpole_balance",
        "cartpole_balance_sparse",
        "cartpole_swingup",
        "cartpole_swingup_sparse",
        "cheetah_run",
        "cup_catch",
        "finger_spin",
        "finger_turn_easy",
        "finger_turn_hard",
        "hopper_hop",
        "hopper_stand",
        "pendulum_swingup",
        "quadruped_run",
        "quadruped_walk",
        "reacher_easy",
        "reacher_hard",
        "walker_run",
        "walker_stand",
        "walker_walk",
    ]
    # Remove the "dm_control/" prefix
    name = env_name.split("/")[1]
    # Remove -v0 if it exists
    name = name.split("-v0")[0]
    # Split into domain and task
    domain, task = name.split("-")
    domain_task = f"{domain}_{task}"
    assert (
        domain_task in datasets
    ), f"Environment {domain_task} not found in the DeepMind Control Suite dataset. \
            Available environments: {datasets}"
    return DeepMindControlSuiteDataset(
        name=domain_task,
        split=split,
        cache_dir=cache_dir,
        num_proc=8,
        block_size=block_size,
        columns=["observation", "action"],
        transform=transform,
    )

def get_masked_distracting_control_suite_dataset(
    env_name: str,
    split: str,
    cache_dir: Optional[str] = None,
    block_size: Optional[int] = None,
    transform: Optional[DictTransform] = None,
) -> MaskedDistractingControlSuiteDataset:
    """Get the available Masked Distracting Control Suite dataset.

    Args:
        name (str): The name of the dataset.
        split (str): The dataset split, e.g., 'train' or 'test'.
        cache_dir (Optional[str]): Directory to cache the dataset.
        block_size (Optional[int]): The block size for the dataset.
        transform (Optional[DictTransform]): The transformation to apply to the dataset.

    Returns:
        Dataset: Available Masked Distracting Control Suite dataset.
    """

    datasets = [
        "cheetah_run", "cheetah_run_distractor_low", "cheetah_run_distractor_hard",
        "hopper_hop", "hopper_hop_distractor_low", "hopper_hop_distractor_hard",
        "humanoid_walk", "humanoid_walk_distractor_low", "humanoid_walk_distractor_hard",
        "walker_run", "walker_run_distractor_low", "walker_run_distractor_hard",
    ]

    # 1. Strip the namespace (e.g., 'dm_control/masked-cheetah-run-v0' -> 'masked-cheetah-run-v0')
    name = env_name.split("/")[-1]

    # 2. Remove the version suffix (e.g., 'masked-cheetah-run-v0' -> 'masked-cheetah-run')
    name = name.split("-v")[0]


    # 3. Handle the 'masked-' prefix and standardize separators
    if name.startswith("masked-"):
        # Remove 'masked-' and keep the rest as is
        name = name.replace("masked-", "")

    # Replace "-" with "_"
    name = name.replace("-", "_")

    # 4. Final Validation
    assert name in datasets, (
        f"Parsed name '{name}' (from {env_name}) not found in dataset list. "
        f"Available: {datasets}"
    )


    # Return your dataset class (ensure the class is imported/defined)
    return MaskedDistractingControlSuiteDataset(
        name=name,
        split=split,
        cache_dir=cache_dir,
        num_proc=8,
        block_size=block_size,
        columns=["observation", "mask", "action"],
        transform=transform,
    )



def get_metaworld_dataset(
    env_name: str,
    split: str,
    cache_dir: Optional[str] = None,
    block_size: Optional[int] = None,
    transform: Optional[DictTransform] = None,
    dataset_path: Optional[str] = None,
) -> MetaWorldDataset:
    """Get the available Meta-World dataset (handles both distracted and vanilla).
    Args:
        env_name (str): The name of the dataset.
        split (str): The dataset split, e.g., 'train' or 'test'.
        cache_dir (Optional[str]): Directory to cache the dataset.
        block_size (Optional[int]): The block size for the dataset.
        transform (Optional[DictTransform]): The transformation to apply to the dataset.
    Returns:
        MetaWorldDataset: Available Meta-World dataset.
    """

    # Parse: Meta-World/masked-MT1-basketball-v3 -> masked-MT1-basketball-v3
    name = env_name.split("/")[-1] if "/" in env_name else env_name

    # Remove MT1- prefix (not part of HuggingFace dataset name)
    assert "MT1-" in name, f"Environment {name} not found in Meta-World dataset. Available environments: {datasets}"
    name = name.replace("MT1-", "")

    # Detect variant BEFORE stripping the prefix from the name, because HuggingFace
    # configs don't have "masked-"/"distracting-" in their names.
    use_distracted_obs = "distracting" in name.lower() or "masked" in name.lower()

    # Now strip prefixes so we get the bare HuggingFace config name (e.g. "basketball-v3")
    name = name.replace("distracting-", "").replace("masked-", "")

    # Dataset source: explicit arg > DMW_DATASET_PATH env var > MetaWorldDataset default.
    # DMW_DATASET_PATH may be a HF repo id (e.g. tsakman23/visual_masked_distracting_metaworld)
    # or a local staged save_to_disk root (<root>/<config>/<split>), so OCL can train on the
    # same DMW data as IM-LAM/MaskLAM without editing the configs.
    if dataset_path is None:
        dataset_path = os.environ.get("DMW_DATASET_PATH")
    path_kwargs = {"path": dataset_path} if dataset_path else {}
    return MetaWorldDataset(
        name=name,  # bare HF config name, e.g. "basketball-v3"
        split=split,
        cache_dir=cache_dir,
        num_proc=8,
        block_size=block_size,
        columns=["observation", "mask", "action"],
        transform=transform,
        use_distracted_obs=use_distracted_obs,
        **path_kwargs,
    )


def get_atari_dataset(
    env_name: str,
    split: str,
    cache_dir: Optional[str] = None,
    block_size: Optional[int] = None,
    transform: Optional[DictTransform] = None,
) -> Union[MaskedFroggerDataset, MaskedFrostbiteDataset]:
    """Get a masked Frogger or Frostbite dataset from Hugging Face."""
    game_name = env_name.split("/")[-1]
    kwargs = {
        "split": split,
        "cache_dir": cache_dir,
        "num_proc": 8,
        "block_size": block_size,
        "columns": ["observation", "mask", "action"],
        "transform": transform,
    }
    if "Frogger" in game_name:
        return MaskedFroggerDataset(**kwargs)
    if "Frostbite" in game_name:
        return MaskedFrostbiteDataset(**kwargs)
    raise NotImplementedError(f"Atari dataset is not supported: {env_name}")


def get_dataset(
    name: str,
    split: str,
    cache_dir: Optional[str] = None,
    block_size: Optional[int] = None,
    **kwargs,
) -> Dataset:
    """
    Get the dataset for the specified environment and split and transform.

    Args:
        name (str): The name of the environment.
        split (str): The dataset split, e.g., 'train' or 'test'.
        cache_dir (Optional[str]): Directory to cache the dataset.
        block_size (Optional[int]): The block size for the dataset.
        **kwargs: Additional arguments for the dataset.

    Returns:
        Dataset: The dataset object.
    """
    transform = get_dataset_transform(name)
    if "procgen" in name:
        return get_procgen_dataset(name, split, cache_dir, block_size, transform)
    elif "MineRL" in name:
        ValueError("MineRL dataset is not supported yet.")
    elif "dm_control" in name:
        if "distracting" in name:
            ValueError("Distracting DMC dataset is not supported yet.")
        elif "masked" in name:
            return get_masked_distracting_control_suite_dataset(name, split, cache_dir, block_size, transform)
        else:
            return get_dmc_dataset(name, split, cache_dir, block_size, transform)
    elif "Meta-World" in name:
        return get_metaworld_dataset(name, split, cache_dir, block_size, transform)
    elif "ALE" in name or "atari" in name:
        return get_atari_dataset(name, split, cache_dir, block_size, transform)
    else:
        raise NotImplementedError("Dataset is currently not supported.")
