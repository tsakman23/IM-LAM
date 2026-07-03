from ifo.common.utils.imports import _IS_VIZDOOM_AVAILABLE

if not _IS_VIZDOOM_AVAILABLE:
    raise ModuleNotFoundError(_IS_VIZDOOM_AVAILABLE)

import gymnasium as gym
from gymnasium.utils import EzPickle
from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv


class VizdoomScenarioEnv(VizdoomEnv, EzPickle):
    """Basic ViZDoom environments which reside in the `scenarios` directory"""

    def __init__(self, scenario_file, frame_skip=1, max_buttons_pressed=1, render_mode=None):
        EzPickle.__init__(self, scenario_file, frame_skip, max_buttons_pressed, render_mode)
        super().__init__(
            scenario_file,
            frame_skip,
            max_buttons_pressed,
            render_mode,
        )


def register_vizdoom_envs() -> None:
    vizdoom_registry = {
        "VizdoomBasic-v1": {"scenario_file": "ifo/common/env/vizdoom_scenarios/basic.cfg"},
        "VizdoomCorridor-v1": {"scenario_file": "ifo/common/env/vizdoom_scenarios/deadly_corridor.cfg"},
        "VizdoomDefendCenter-v1": {"scenario_file": "ifo/common/env/vizdoom_scenarios/defend_the_center.cfg"},
        "VizdoomDefendLine-v1": {"scenario_file": "ifo/common/env/vizdoom_scenarios/defend_the_line.cfg"},
        "VizdoomHealthGathering-v1": {"scenario_file": "ifo/common/env/vizdoom_scenarios/health_gathering.cfg"},
        "VizdoomMyWayHome-v1": {"scenario_file": "ifo/common/env/vizdoom_scenarios/my_way_home.cfg"},
        "VizdoomPredictPosition-v1": {"scenario_file": "ifo/common/env/vizdoom_scenarios/predict_position.cfg"},
        "VizdoomTakeCover-v1": {"scenario_file": "ifo/common/env/vizdoom_scenarios/take_cover.cfg"},
        "VizdoomDeathmatch-v1": {"scenario_file": "ifo/common/env/vizdoom_scenarios/deathmatch.cfg"},
        "VizdoomHealthGatheringSupreme-v1": {
            "scenario_file": "ifo/common/env/vizdoom_scenarios/health_gathering_supreme.cfg"
        },
    }

    for id, kwargs in vizdoom_registry.items():
        if id not in gym.envs.registry:
            gym.register(id=id, entry_point=VizdoomScenarioEnv, kwargs=kwargs)
