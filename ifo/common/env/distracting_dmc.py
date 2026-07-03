from ifo.common.utils.imports import _IS_DM_CONTROL_AVAILABLE

if not _IS_DM_CONTROL_AVAILABLE:
    raise ModuleNotFoundError(_IS_DM_CONTROL_AVAILABLE)

from functools import partial

from gymnasium.envs.registration import register
from shimmy.utils.envs_configs import DM_CONTROL_SUITE_ENVS

from ifo.common.env.dcs import suite_utils
from ifo.common.env.dcs.suite import get_dcs_env

MASKED_DISTRACTING_CONTROL_SUITE_ENV_DIFFICULTY_MODES= {
            "low":  {"scale": "vanilla", "video": "hard", "dyn": True},
            "hard": {"scale": "easy", "video": "hard", "dyn": True}
        }

def register_distracting_dmc_envs() -> None:
    """Register Distracting Control Suite environments according to https://arxiv.org/abs/2502.00379"""
    for _domain_name, _task_name in DM_CONTROL_SUITE_ENVS:
        # Vanilla/clean DMC environments (no distractors, no camera scale changes)
        register(
            id=f"dm_control/masked-{_domain_name}-{_task_name}-v0",
            entry_point=partial(
                get_dcs_env,
                domain_name=_domain_name,
                task_name=_task_name,
                difficulty_video="vanilla",
                difficulty_scale="vanilla",
                dynamic=False,
                background_dataset_videos=None,
                render_kwargs=dict(height=64, width=64),
            ),
        )

        for mode_name, settings in MASKED_DISTRACTING_CONTROL_SUITE_ENV_DIFFICULTY_MODES.items():
            register(
                id=f"dm_control/masked-{_domain_name}-{_task_name}-distractor-{mode_name}-v0",
                entry_point=partial(
                    get_dcs_env,
                    domain_name=_domain_name,
                    task_name=_task_name,
                    difficulty_video=settings["video"],
                    difficulty_scale=settings["scale"],
                    dynamic=settings["dyn"],
                    background_dataset_videos="train",
                    render_kwargs=dict(height=64, width=64),
                ),
            )

        # Register DCS environments according to distractor difficulties defined in LAOM paper: https://arxiv.org/abs/2502.00379
        for difficulty_video in suite_utils.DIFFICULTY_NUM_VIDEOS.keys():
            for difficulty_scale in suite_utils.DIFFICULTY_SCALE.keys():
                register(
                    id=f"dm_control/{_domain_name}-{_task_name}-distracting-scale-{difficulty_scale}-video-{difficulty_video}-v0",
                    entry_point=partial(
                        get_dcs_env,
                        domain_name=_domain_name,
                        task_name=_task_name,
                        difficulty_video=difficulty_video,
                        difficulty_scale=difficulty_scale,
                        dynamic=True,
                        background_dataset_videos="train",
                        render_kwargs=dict(height=64, width=64),
                    ),
                )
