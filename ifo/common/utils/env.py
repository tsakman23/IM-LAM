import multiprocessing
import os
import zipfile
from functools import partial
from typing import Literal, Optional

import gymnasium as gym
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.wrappers import AtariPreprocessing
from pypdl import Pypdl

from ifo.common.env.wrapper import (
    ActionRepeat,
    FilterObservation,
    MetaWorldImageWrapper,
    MetaworldTerminateOnSuccessWrapper,
    OriginalWrapper,
    TransposeObservation,
    VectorFrameStackObservation,
    VecVideoRecorder,
)
from ifo.common.transforms.env import get_env_transform
from ifo.common.utils.imports import (
    _IS_ATARI_AVAILABLE,
    _IS_DM_CONTROL_AVAILABLE,
    _IS_METAWORLD_AVAILABLE,
    _IS_MINERL_AVAILABLE,
    _IS_MINIGRID_AVAILABLE,
    _IS_PROCGEN_AVAILABLE,
    _IS_VIZDOOM_AVAILABLE,
)
from ifo.common.utils.render import vec_render


def download_davis_dataset(background_dataset_path: str) -> None:
    """
    Download and extract the DAVIS 2017 dataset if not already present.

    Args:
        background_dataset_path (str): Path to the DAVIS directory that contains the video directories.
    """
    if os.path.exists(background_dataset_path):
        return

    print("Downloading DAVIS dataset...")
    os.makedirs(background_dataset_path, exist_ok=True)
    base_path = background_dataset_path[:-5]
    dl = Pypdl()
    dl.start(
        url="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip", file_path=base_path
    )
    # Unzip file.
    zip_file_path = base_path + "DAVIS-2017-trainval-480p.zip"
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(base_path)
    # Remove zip file.
    os.remove(zip_file_path)


def make_metaworld_env(
    env_name: str,
    num_envs: int = 1,
    action_repeat: int = 1,
    render_mode: str = "rgb_array",
    camera_name: Literal["corner", "corner2", "corner3", "topview", "behindGripper", "gripperPOV"] = "corner",
    background_dataset_path: str = "datasets/DAVIS",
    distracting: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> VectorEnv:
    """
    Create a Meta-World environment.
    """
    assert "Meta-World" in env_name, "Environment name must contain 'Meta-World' to create a MetaWorld environment."
    if _IS_METAWORLD_AVAILABLE:
        import metaworld

        from ifo.common.env.dmw.wrapper import DistractingMetaworldWrapper
    else:
        raise ModuleNotFoundError("Meta-World is not installed")

    # Check if environment name contains a postfixed task name, e.g. "Meta-World/MT1-reach-v3" or "Meta-World/ML1-push-v3".
    contains_task = [env_name.endswith(task) for task in metaworld.ALL_V3_ENVIRONMENTS.keys()]
    if any(contains_task):
        task_name = list(metaworld.ALL_V3_ENVIRONMENTS.keys())[contains_task.index(True)]
        env_name = env_name.replace(f"-{task_name}", "")
    else:
        task_name = None

    if "ML" in env_name:
        raise ValueError("ML environments are bugged in 3.0.0, use MT environments instead.")
    if any(x in env_name for x in ("MT10", "MT25", "MT50")):
        required_num_envs = int(env_name.split("/")[-1])
        assert num_envs == required_num_envs, f"MT environments must be created with {required_num_envs} environments."

    wrappers = [
        partial(MetaWorldImageWrapper, auto_rotate=not distracting),
        partial(FilterObservation, key="image"),
        MetaworldTerminateOnSuccessWrapper,
    ]
    if action_repeat > 1:
        wrappers.append(partial(ActionRepeat, repeat=action_repeat))

    if distracting:
        download_davis_dataset(background_dataset_path)
        wrappers.insert(0, partial(DistractingMetaworldWrapper, dataset_path=f"{background_dataset_path}/JPEGImages/480p"))

    env_kwargs = {
        "wrappers": wrappers,
        "render_mode": "rgb_array",
        "camera_name": camera_name,
        "width": 128,
        "height": 128,
        "vector_strategy": "async",
        "autoreset_mode": AutoresetMode.NEXT_STEP,
        "seed": seed,
    }
    if task_name:
        env_kwargs["num_envs"] = num_envs
        env_kwargs["env_name"] = task_name
        env_kwargs["disable_env_checker"] = True
        del env_kwargs["vector_strategy"]
        env_kwargs["vectorization_mode"] = "async"

    env = gym.make_vec(
        env_name,
        **env_kwargs,
    )
    # Set custom render function to get tiled observation render.
    env.render = lambda: vec_render(self=env, mode=render_mode)
    return env


def make_atari_env(
    env_name: str,
    num_envs: int = 1,
    action_repeat: int = 1,
    render_mode: str = "rgb_array",
    screen_size: int = 96,
    grayscale_obs: bool = True,
    sticky_actions: float = 0.25,
    **kwargs,
) -> VectorEnv:
    """
    Create a Atari environment.

    We overwrite the default Atari settings with our sticky_actions and action_repeat.

    Note: To create previously implemented environment use the following parameters,
    gymnasium.make(env_id, obs_type=..., frameskip=..., repeat_action_probability=..., full_action_space=...).

    See: https://ale.farama.org/environments/#version-history-and-naming-schemes

    Args:
        env_name (str): Environment name.
        num_envs (int): Number of environments to create.
        action_repeat (int): Number of times to repeat each action.
        render_mode (str): Render mode.
        screen_size (int): Size of the screen.
        grayscale_obs (bool): Whether to use grayscale observations.
        sticky_actions (float): Probability of repeating the same action.
        **kwargs: Additional keyword arguments.

    Returns:
        VectorEnv: Wrapped vectorized gym environment.
    """
    assert "ALE" in env_name, "Environment name must contain 'ALE' to create a Atari environment."
    if _IS_ATARI_AVAILABLE:
        import ale_py  # noqa: F401
    else:
        raise ModuleNotFoundError("Atari is not installed")

    env = gym.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[
            partial(
                AtariPreprocessing,
                frame_skip=action_repeat,
                screen_size=screen_size,
                grayscale_obs=grayscale_obs,
                grayscale_newaxis=True,
            )
        ],
        render_mode=render_mode,
        vectorization_mode="async",
        vector_kwargs=dict(autoreset_mode=AutoresetMode.NEXT_STEP),
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=sticky_actions,
        full_action_space=False,
    )
    env.render = lambda: vec_render(self=env, mode=render_mode)
    return env


def make_procgen_env(env_name: str, num_envs: int = 1, action_repeat: int = 1, **kwargs) -> VectorEnv:
    """
    Create a Procgen environment.

    Args:
        env_name (str): Environment name.
        num_envs (int): Number of environments to create.
        action_repeat (int): Number of times to repeat each action.
        **kwargs: Additional keyword arguments.

    Returns:
        VectorEnv: Wrapped vectorized gym environment.
    """
    assert "procgen" in env_name, "Environment name must contain 'procgen' to create a Procgen environment."
    if _IS_PROCGEN_AVAILABLE:
        from ifo.common.env.procgen import ProcgenEnv  # noqa: F401
    else:
        raise ModuleNotFoundError("Procgen is not installed")

    assert action_repeat == 1, "Action repeat is not supported for Procgen environments."
    # Strip full env name, e.g. 'prcogen-env_name-v0', to base env name, e.g. 'env_name'.
    env_name = env_name.split("-")[1]
    env = ProcgenEnv(
        env_name=env_name,
        num_envs=num_envs,
        distribution_mode="easy",
        num_levels=0,
        start_level=0,
        num_threads=max(4, int(num_envs**0.6)),
    )
    return env


def make_minigrid_env(
    env_name: str, num_envs: int = 1, action_repeat: int = 1, render_mode: str = "rgb_array", **kwargs
) -> VectorEnv:
    """
    Create a MiniGrid environment.

    Args:
        env_name (str): Environment name.
        num_envs (int): Number of environments to create.
        action_repeat (int): Number of times to repeat each action.
        render_mode (str): Render mode.
        **kwargs: Additional keyword arguments.

    Returns:
        VectorEnv: Wrapped vectorized gym environment.
    """
    assert "MiniGrid" in env_name, "Environment name must contain 'MiniGrid' to create a MiniGrid environment."
    if _IS_MINIGRID_AVAILABLE:
        from minigrid.wrappers import ImgObsWrapper  # noqa: F401

        register_minigrid_envs()
    else:
        raise ModuleNotFoundError("MiniGrid is not installed")

    wrappers = [ImgObsWrapper]
    if action_repeat > 1:
        wrappers.append(partial(ActionRepeat, repeat=action_repeat))
    env = gym.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=wrappers,
        render_mode=render_mode,
        vectorization_mode="async",
        autoreset_mode=AutoresetMode.NEXT_STEP,
    )
    # Set custom render function to get tiled observation render.
    env.render = lambda: vec_render(self=env, mode=render_mode)
    return env


def make_vizdoom_env(
    env_name: str, num_envs: int = 1, action_repeat: int = 1, render_mode: str = "rgb_array", **kwargs
) -> VectorEnv:
    """
    Create a Vizdoom environment.

    Args:
        env_name (str): Environment name.
        num_envs (int): Number of environments to create.
        action_repeat (int): Number of times to repeat each action.
        render_mode (str): Render mode.
        **kwargs: Additional keyword arguments.

    Returns:
        VectorEnv: Wrapped vectorized gym environment.
    """
    assert "Vizdoom" in env_name, "Environment name must contain 'vizdoom' to create a Vizdoom environment."
    if _IS_VIZDOOM_AVAILABLE:
        from ifo.common.env.vizdoom import register_vizdoom_envs

        register_vizdoom_envs()
    else:
        raise ModuleNotFoundError("Vizdoom is not installed")

    wrappers = [partial(FilterObservation, key="screen")]
    if action_repeat > 1:
        wrappers.append(partial(ActionRepeat, repeat=action_repeat))
    env = gym.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=wrappers,
        render_mode=render_mode,
        vectorization_mode="async",
        autoreset_mode=AutoresetMode.NEXT_STEP,
    )
    # Use custom render function to get tiled observation render.
    env.render = lambda: vec_render(self=env, mode=render_mode)
    return env


def make_minerl_env(
    env_name: str, num_envs: int = 1, action_repeat: int = 1, render_mode: str = "rgb_array", **kwargs
) -> VectorEnv:
    """
    Create a MineRL environment.

    Args:
        env_name (str): Environment name.
        num_envs (int): Number of environments to create.
        action_repeat (int): Number of times to repeat each action.
        render_mode (str): Render mode.
        **kwargs: Additional keyword arguments.

    Returns:
        VectorEnv: Wrapped vectorized gym environment.
    """
    assert "MineRL" in env_name, "Environment name must contain 'MineRL' to create a MineRL environment."
    if _IS_MINERL_AVAILABLE:
        from ifo.common.env.minerl import register_minerl_envs

        register_minerl_envs()
    else:
        raise ModuleNotFoundError("MineRL is not installed")

    wrappers = [partial(FilterObservation, key="pov")]
    if action_repeat > 1:
        wrappers.append(partial(ActionRepeat, repeat=action_repeat))
    env = gym.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=wrappers,
        render_mode=render_mode,
        vectorization_mode="async",
        disable_env_checker=True,
        autoreset_mode=AutoresetMode.NEXT_STEP,
    )
    # Use custom render function to get tiled observation render.
    env.render = lambda: vec_render(self=env, mode=render_mode)
    return env


def make_dmc_env(
    env_name: str, num_envs: int = 1, action_repeat: int = 1, render_mode: str = "rgb_array", **kwargs
) -> VectorEnv:
    """
    Create a DMC environment.

    Args:
        env_name (str): Name of the environment.
        num_envs (int): Number of environments to create.
        action_repeat (int): Number of times to repeat each action.
        render_mode (str): Render mode.
        **kwargs: Additional keyword arguments.

    Returns:
        VectorEnv: Wrapped vectorized gym environment.
    """
    assert "dm_control" in env_name, "Environment name must contain 'dm_control' to create a DMC environment."
    if _IS_DM_CONTROL_AVAILABLE:
        import shimmy  # noqa: F401
    else:
        raise ModuleNotFoundError("DMC is not installed")

    wrappers = [partial(gym.wrappers.AddRenderObservation, render_only=True)]
    if action_repeat > 1:
        wrappers.append(partial(ActionRepeat, repeat=action_repeat))
    env = gym.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=wrappers,
        render_mode=render_mode,
        vectorization_mode="async",
        vector_kwargs=dict(autoreset_mode=AutoresetMode.NEXT_STEP),
        render_kwargs=dict(camera_id=0, height=64, width=64),
    )
    env.render = lambda: vec_render(self=env, mode=render_mode)
    return env


def make_distracting_dmc_env(
    env_name: str,
    num_envs: int = 1,
    action_repeat: int = 1,
    render_mode: str = "rgb_array",
    background_dataset_path: str = "datasets/DAVIS",
    **kwargs,
) -> VectorEnv:
    """
    Create a Distracting DMC environment.

    Args:
        env_name (str): Name of the environment.
        num_envs (int): Number of environments to create.
        action_repeat (int): Number of times to repeat each action.
        render_mode (str): Render mode.
        background_dataset_path (str): Path to the DAVIS directory that contains the video directories.
        **kwargs: Additional keyword arguments.

    Returns:
        VectorEnv: Wrapped vectorized gym environment.
    """
    assert "dm_control" in env_name, "Environment name must contain 'dm_control' to create a DCS environment."
    assert "distracting" in env_name or "masked" in env_name, "Environment name must contain 'distracting' or 'masked' to create a DCS environment."
    if _IS_DM_CONTROL_AVAILABLE:
        from ifo.common.env.distracting_dmc import register_distracting_dmc_envs

        register_distracting_dmc_envs()
    else:
        raise ModuleNotFoundError("DMC is not installed")

    wrappers = [partial(gym.wrappers.AddRenderObservation, render_only=True)]
    if action_repeat > 1:
        wrappers.append(partial(ActionRepeat, repeat=action_repeat))
    # Download DAVIS dataset, if not already present.
    download_davis_dataset(background_dataset_path)

    env = gym.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=wrappers,
        render_mode=render_mode,
        vectorization_mode="async",
        vector_kwargs=dict(autoreset_mode=AutoresetMode.NEXT_STEP),
        render_kwargs=dict(camera_id=0, height=64, width=64),
        background_dataset_path=f"{background_dataset_path}/JPEGImages/480p",
    )
    env.render = lambda: vec_render(self=env, mode=render_mode)
    return env


def make_env(
    name: str,
    num_envs: int = 1,
    record_video: bool = False,
    block_size: Optional[int] = 1,
    action_repeat: int = 1,
    **kwargs,
) -> VectorEnv:
    """
    Utility function for multiprocessed env.

    Creates a vectorized environment, which is wrapped with the necessary wrappers.

    Args:
        name (str): Environment name.
        num_envs (int): Number of environments to create.
        record_video (bool): Whether to record videos.
        block_size (Optional[int]): Sequence length of observations. If not specified, no block stacking is performed
            and the observation is returned without additional time dimension.
        action_repeat (int): Number of times to repeat each action.
        **kwargs: Additional keyword arguments.

    Returns:
        VectorEnv: Wrapped vectorized gym environment.
    """
    if "dm_control" in name:
        # Set multiprocessing start method to "spawn" to avoid issues with rendering in async mode.
        multiprocessing.set_start_method("spawn", force=True)
        # Create the Distracting DMC environment.
        if "distracting" in name or "masked" in name:
            name = name.replace("sam-", "")  # Remove SAM prefix if it exists
            env = make_distracting_dmc_env(name, num_envs, action_repeat, **kwargs)
        else:
            env = make_dmc_env(name, num_envs, action_repeat, **kwargs)
    elif "Meta-World" in name:
        # Set multiprocessing start method to "spawn" to avoid issues with rendering in async mode.
        multiprocessing.set_start_method("spawn", force=True)
        name = name.replace("sam-", "")  # Remove SAM prefix if it exists
        if "distracting" in name or "masked" in name:
            name = name.replace("distracting-", "").replace("masked-", "")
            env = make_metaworld_env(name, num_envs, action_repeat, distracting=True, **kwargs)
        else:
            env = make_metaworld_env(name, num_envs, action_repeat, **kwargs)
    else:
        raise NotImplementedError(f"Environment is currently not supported for this submission. Name: {name}")

    # Record video if required.
    if record_video:
        required_kwargs = ["video_folder", "step_trigger", "video_length", "run_id"]
        for kwarg in required_kwargs:
            assert kwarg in kwargs, f"Keyword argument {kwarg} is required for recording videos."
        env = VecVideoRecorder(
            env,
            video_folder=f"{kwargs['video_folder']}/{kwargs['run_id']}",
            step_trigger=lambda x: (x % (int(kwargs["step_trigger"] / env.num_envs / action_repeat))) == 0,
            video_length=int(kwargs["video_length"]),
        )

    # Record episode statistics for correct metrics.
    env = gym.wrappers.vector.RecordEpisodeStatistics(env, stats_key="episode_statistics")
    # Wrap the environment to preserve the original environment before modifications.
    env = OriginalWrapper(env) if kwargs.get("preserve_original", False) else env

    # Apply custom transformations.
    transform = get_env_transform(name)
    if transform is not None:
        env = transform(env)

    # Add sequence length (history) to observations.
    if block_size:
        env = VectorFrameStackObservation(env, block_size, padding_type="zero")

    # (T, B, H, W, C) -> (B, T, C, H, W) or (B, H, W, C) -> (B, C, H, W)
    permutation = (1, 0, 4, 2, 3) if block_size else (0, 3, 1, 2)
    env = TransposeObservation(env, permutation=permutation)
    env.action_repeat = action_repeat

    return env


