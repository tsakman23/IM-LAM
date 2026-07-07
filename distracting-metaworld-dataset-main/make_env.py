import os
import zipfile
from functools import partial
from typing import Literal, Optional

import gymnasium as gym
from gymnasium.vector import AutoresetMode, VectorEnv
from pypdl import Pypdl

from env.wrapper import (
    ActionRepeat,
    MetaWorldImageWrapper,
    MetaworldTerminateOnSuccessWrapper,
)


def download_davis_dataset(background_dataset_path: str) -> None:
    """Lädt das DAVIS-Dataset für Distracting-Umgebungen herunter."""
    if os.path.exists(background_dataset_path):
        return

    print("Downloading DAVIS dataset...")
    os.makedirs(background_dataset_path, exist_ok=True)
    base_path = background_dataset_path[:-5]
    dl = Pypdl()
    dl.start(
        url="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip", 
        file_path=base_path
    )
    zip_file_path = base_path + "DAVIS-2017-trainval-480p.zip"
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(base_path)
    os.remove(zip_file_path)


def make_metaworld_env(
    env_name: str,
    num_envs: int = 1,
    action_repeat: int = 1,
    render_mode: str = "rgb_array",
    camera_name: Literal["corner", "corner2", "corner3", "topview", "behindGripper", "gripperPOV"] = "corner",
    background_dataset_path: str = "datasets/DAVIS",
    distracting: bool = False,
    segmentation: bool = False,
    seed: Optional[int] = None,
    dataset_videos: Optional[str] = "train",
    object_body_names: Optional[list] = None,
    **kwargs,
) -> VectorEnv:
    """Erstellt eine Meta-World Umgebung."""
    import metaworld

    from env.dmw.wrapper import DistractingMetaworldWrapper, SegmentationMetaworldWrapper

    # Task-Namen Mapping
    contains_task = [env_name.endswith(task) for task in metaworld.ALL_V3_ENVIRONMENTS.keys()]
    if any(contains_task):
        task_name = list(metaworld.ALL_V3_ENVIRONMENTS.keys())[contains_task.index(True)]
        env_name = env_name.replace(f"-{task_name}", "")
    else:
        task_name = None

    if "ML" in env_name:
        raise ValueError("ML environments are bugged in 3.0.0, use MT environments instead.")

    wrappers = [
        partial(MetaWorldImageWrapper, auto_rotate=not distracting and not segmentation),
        MetaworldTerminateOnSuccessWrapper,
    ]
    if action_repeat > 1:
        wrappers.append(partial(ActionRepeat, repeat=action_repeat))

    if distracting:
        download_davis_dataset(background_dataset_path)
        wrappers.insert(0, partial(DistractingMetaworldWrapper, dataset_path=f"{background_dataset_path}/JPEGImages/480p", dataset_videos=dataset_videos))
    
    if segmentation:
        wrappers.insert(0, partial(SegmentationMetaworldWrapper, object_body_names=object_body_names))

    env_kwargs = {
        "wrappers": wrappers,
        "render_mode": "rgb_array",
        "camera_name": camera_name,
        "width": kwargs.pop("width", 128),
        "height": kwargs.pop("height", 128),
        "vector_strategy": "async",
        "autoreset_mode": AutoresetMode.NEXT_STEP,
        "seed": seed,
    }

    if task_name:
        env_kwargs["num_envs"] = num_envs
        env_kwargs["env_name"] = task_name
        env_kwargs["disable_env_checker"] = True
        del env_kwargs["vector_strategy"]
        env_kwargs["vectorization_mode"] = "async" if num_envs > 1 else "sync"

    env = gym.make_vec(env_name, **env_kwargs)
    return env
