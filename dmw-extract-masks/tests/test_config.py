import json
import os

from utils import resolve_task_config

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_REQUIRED_GLOBAL_KEYS = [
    "source_dataset_id", "grounding_dino_model_id", "sam2_model_id",
    "observation_column", "box_threshold", "text_threshold",
    "fallback_box_threshold", "fallback_text_threshold",
]


def _load_config():
    with open(os.path.join(_PKG, "config.json")) as f:
        return json.load(f)


def test_shipped_config_resolves_push_v3_with_required_keys():
    resolved = resolve_task_config(_load_config(), "push-v3")
    assert isinstance(resolved["agent_bbox"], list) and len(resolved["agent_bbox"]) == 4
    assert isinstance(resolved["object_prompt"], str) and resolved["object_prompt"].strip()
    for key in _REQUIRED_GLOBAL_KEYS:
        assert key in resolved, f"missing required config key: {key}"


def test_observation_column_is_distracted():
    # DMW is distracting Meta-World: masks are computed on the distracted frames.
    resolved = resolve_task_config(_load_config(), "push-v3")
    assert resolved["observation_column"] == "observation_distracted"
