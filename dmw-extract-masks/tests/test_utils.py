import pytest
import torch

from utils import get_torch_dtype, resolve_task_config


def test_get_torch_dtype_maps_known():
    assert get_torch_dtype("bfloat16") is torch.bfloat16
    assert get_torch_dtype("float16") is torch.float16
    assert get_torch_dtype("float32") is torch.float32


def test_get_torch_dtype_rejects_unknown():
    with pytest.raises(ValueError):
        get_torch_dtype("float64")


def test_resolve_task_config_merges_global_defaults_into_task():
    config = {
        "box_threshold": 0.3,
        "text_threshold": 0.25,
        "sam2_model_id": "facebook/sam2-hiera-tiny",
        "tasks": {
            "push-v3": {"agent_bbox": [1, 2, 3, 4], "object_prompt": "red puck."},
        },
    }
    resolved = resolve_task_config(config, "push-v3")
    # Global scalar defaults are carried down...
    assert resolved["box_threshold"] == 0.3
    assert resolved["text_threshold"] == 0.25
    assert resolved["sam2_model_id"] == "facebook/sam2-hiera-tiny"
    # ...alongside the per-task fields.
    assert resolved["agent_bbox"] == [1, 2, 3, 4]
    assert resolved["object_prompt"] == "red puck."
    # The nested "tasks" table itself is not leaked into the resolved config.
    assert "tasks" not in resolved


def test_resolve_task_config_task_overrides_global():
    config = {
        "box_threshold": 0.3,
        "tasks": {"push-v3": {"agent_bbox": [0, 0, 1, 1], "box_threshold": 0.45}},
    }
    resolved = resolve_task_config(config, "push-v3")
    assert resolved["box_threshold"] == 0.45


def test_resolve_task_config_missing_task_raises():
    config = {"tasks": {"push-v3": {}}}
    with pytest.raises(KeyError):
        resolve_task_config(config, "nonexistent-v3")
