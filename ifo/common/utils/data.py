from typing import Literal, Optional

from torch.utils.data import Dataset

from ifo.common.data import (
    DeepMindControlSuiteDataset,
    DistractingControlSuiteDataset,
    MaskedDistractingControlSuiteDataset,
    MetaWorldDataset,
    MineRLBasaltDataset,
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


def get_minerl_dataset(
    env_name: str,
    split: str,
    cache_dir: Optional[str] = None,
    block_size: Optional[int] = None,
    transform: Optional[DictTransform] = None,
) -> MineRLBasaltDataset:
    """Get the available MineRL dataset.

    Args:
        env_name (str): The name of the environment.
        split (str): The dataset split, e.g., 'train' or 'test'.
        cache_dir (Optional[str]): Directory to cache the dataset.
        block_size (Optional[int]): The block size for the dataset.
        transform (Optional[DictTransform]): The transformation to apply to the dataset.

    Returns:
        MineRLBasaltDataset: Available MineRL dataset.
    """
    environment_task_name = {
        "MineRLBasaltFindCave-v0": "FindCave",
        "MineRLBasaltMakeWaterfall-v0": "MakeWaterfall",
        "MineRLBasaltCreateVillageAnimalPen-v0": "MakeVillageAnimalPen",
        "MineRLBasaltBuildVillageHouse-v0": "BuildVillageHouse",
        "MineRLTreechop-v0": "SurvivalEarlyGameData",
        "MineRLObtainDiamondPickaxe-v0": "ObtainDiamondPickaxe",
    }
    assert (
        env_name in environment_task_name.keys()
    ), f"Environment {env_name} not found in the MineRL dataset. Available environments: {environment_task_name.keys()}"
    return MineRLBasaltDataset(
        task_names=environment_task_name[env_name],
        split=split,
        cache_dir=cache_dir,
        num_proc=8,
        block_size=block_size,
        transform=transform,
        num_parallel_videos=16,
    )


def get_distracting_control_suite_dataset(
    name: str,
    split: str,
    cache_dir: Optional[str] = None,
    block_size: Optional[int] = None,
    transform: Optional[DictTransform] = None,
    labeled: bool = False,
    labeled_size: Optional[Literal[2, 4, 8, 16, 32, 64, 128]] = None,
) -> Optional[DistractingControlSuiteDataset]:
    """Get the available Distracting Control Suite dataset.

    Args:
        name (str): The name of the dataset.
        split (str): The dataset split, e.g., 'train' or 'test'.
        cache_dir (Optional[str]): Directory to cache the dataset.
        block_size (Optional[int]): The block size for the dataset.
        transform (Optional[DictTransform]): The transformation to apply to the dataset.
        labeled (bool): Whether the labeled dataset is used.
        labeled_size (Optional[Literal[2, 4, 8, 16, 32, 64, 128]]): The size of the labeled dataset.

    Returns:
        Optional[DistractingControlSuiteDataset]: Available Distracting Control Suite dataset.
            If the dataset is not available, returns None.
    """
    env_name_to_dataset_name_train = {
        # Distracting
        "dm_control/cheetah-run-distracting-scale-easy-video-hard-v0": "cheetah-run-scale-easy-video-hard-64px-5k",
        "dm_control/walker-run-distracting-scale-easy-video-hard-v0": "walker-run-scale-easy-video-hard-64px-5k",
        "dm_control/hopper-hop-distracting-scale-easy-video-hard-v0": "hopper-hop-scale-easy-video-hard-64px-5k",
        "dm_control/humanoid-walk-distracting-scale-easy-video-hard-v0": "humanoid-walk-scale-easy-video-hard-64px-5k",
        # Vanilla
        "dm_control/cheetah-run-distracting-scale-vanilla-video-vanilla-v0": "cheetah-run-vanilla-64px-5k",
        "dm_control/walker-run-distracting-scale-vanilla-video-vanilla-v0": "walker-run-vanilla-64px-5k",
        "dm_control/hopper-hop-distracting-scale-vanilla-video-vanilla-v0": "hopper-hop-vanilla-64px-5k",
        "dm_control/humanoid-walk-distracting-scale-vanilla-video-vanilla-v0": "humanoid-walk-vanilla-64px-5k",
    }
    env_name_to_dataset_name_test = {
        # Distracting
        "dm_control/cheetah-run-distracting-scale-easy-video-hard-v0": "cheetah-run-scale-easy-video-hard-64px-eval",
        "dm_control/walker-run-distracting-scale-easy-video-hard-v0": "walker-run-scale-easy-video-hard-64px-eval",
        "dm_control/hopper-hop-distracting-scale-easy-video-hard-v0": "hopper-hop-scale-easy-video-hard-64px-eval",
        "dm_control/humanoid-walk-distracting-scale-easy-video-hard-v0": "humanoid-walk-scale-easy-video-hard-64px-eval",
        # Vanilla. Not available.
        "dm_control/cheetah-run-distracting-scale-vanilla-video-vanilla-v0": "cheetah-run-vanilla-labeled-1000x128-64px",
        "dm_control/walker-run-distracting-scale-vanilla-video-vanilla-v0": "walker-run-vanilla-labeled-1000x128-64px",
        "dm_control/hopper-hop-distracting-scale-vanilla-video-vanilla-v0": "hopper-hop-vanilla-labeled-1000x128-64px",
        "dm_control/humanoid-walk-distracting-scale-vanilla-video-vanilla-v0": "humanoid-walk-vanilla-labeled-1000x128-64px",
    }
    env_name_to_dataset_name_train_labeled = {
        # Distracting
        "dm_control/cheetah-run-distracting-scale-easy-video-hard-v0": "cheetah-run-scale-easy-video-hard-labeled-1000",
        "dm_control/walker-run-distracting-scale-easy-video-hard-v0": "walker-run-scale-easy-video-hard-labeled-1000",
        "dm_control/hopper-hop-distracting-scale-easy-video-hard-v0": "hopper-hop-scale-easy-video-hard-labeled-1000",
        "dm_control/humanoid-walk-distracting-scale-easy-video-hard-v0": "humanoid-walk-scale-easy-video-hard-labeled-1000",
        # Vanilla
        "dm_control/cheetah-run-distracting-scale-vanilla-video-vanilla-v0": "cheetah-run-vanilla-labeled-1000",
        "dm_control/walker-run-distracting-scale-vanilla-video-vanilla-v0": "walker-run-vanilla-labeled-1000",
        "dm_control/hopper-hop-distracting-scale-vanilla-video-vanilla-v0": "hopper-hop-vanilla-labeled-1000",
        "dm_control/humanoid-walk-distracting-scale-vanilla-video-vanilla-v0": "humanoid-walk-vanilla-labeled-1000",
    }
    env_name_to_dataset_name = env_name_to_dataset_name_train if split == "train" else env_name_to_dataset_name_test
    assert (
        name in env_name_to_dataset_name.keys()
    ), f"Environment {name} not found in the Distracting Control Suite dataset. Available environments: {env_name_to_dataset_name.keys()}"
    if labeled:
        possible_labeled_sizes = [2, 4, 8, 16, 32, 64, 128]
        assert labeled_size in possible_labeled_sizes, f"Labeled size should be one of: {possible_labeled_sizes}"
        assert split == "train", "Labeled dataset is only available for train split."
        dataset_name = f"{env_name_to_dataset_name_train_labeled[name]}x{labeled_size}-64px"
    else:
        dataset_name = env_name_to_dataset_name[name]
    if dataset_name is None:
        print(f"Warning: Dataset {name} is not available for split {split}. Returning None.")
        return None
    return DistractingControlSuiteDataset(
        name=dataset_name,
        cache_dir=cache_dir,
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

    # Strip the namespace (e.g., 'dm_control/masked-cheetah-run-v0' -> 'masked-cheetah-run-v0')
    name = env_name.split("/")[-1]

    # Remove the version suffix (e.g., 'masked-cheetah-run-v0' -> 'masked-cheetah-run')
    name = name.split("-v")[0]

    # Check if the dataset should be using SAM masks
    use_sam_mask = False
    if name.startswith("sam-"):
        use_sam_mask = True
        name = name.replace("sam-", "")


    # Handle the 'masked-' prefix and standardize separators
    if name.startswith("masked-"):
        # Remove 'masked-' and keep the rest as is
        name = name.replace("masked-", "")

    # Replace "-" with "_"
    name = name.replace("-", "_")

    # Final Validation
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
        use_sam_mask=use_sam_mask,
    )



def get_metaworld_dataset(
    env_name: str,
    split: str,
    cache_dir: Optional[str] = None,
    block_size: Optional[int] = None,
    transform: Optional[DictTransform] = None,
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
    assert "MT1-" in name, f"Only MT1- environments are supported. Got {name}"
    name = name.replace("MT1-", "")

    # Detect variant BEFORE stripping the prefix from the name, because HuggingFace
    # configs don't have "masked-"/"distracting-" in their names.
    use_distracted_obs = "distracting" in name.lower() or "masked" in name.lower()

    # Now strip prefixes so we get the bare HuggingFace config name (e.g. "basketball-v3")
    name = name.replace("distracting-", "").replace("masked-", "")

    # Check if the dataset should be using SAM masks
    use_sam_mask = False
    if name.startswith("sam-"):
        use_sam_mask = True
        name = name.replace("sam-", "")

    return MetaWorldDataset(
        name=name,  # bare HF config name, e.g. "basketball-v3"
        split=split,
        cache_dir=cache_dir,
        num_proc=8,
        block_size=block_size,
        columns=["observation", "mask", "action"],
        transform=transform,
        use_distracted_obs=use_distracted_obs,
        use_sam_mask=use_sam_mask,
    )

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
    if "dm_control" in name:
        if "masked" in name:
            return get_masked_distracting_control_suite_dataset(name, split, cache_dir, block_size, transform)
        else:
            return get_dmc_dataset(name, split, cache_dir, block_size, transform)
    elif "Meta-World" in name:
        return get_metaworld_dataset(name, split, cache_dir, block_size, transform)
    else:
        raise NotImplementedError("Dataset is currently not supported for this submission.")
