"""Tests for local_save.py - persisting a generated dataset in a training-loadable layout.

Kept free of metaworld/mujoco imports so it runs fast and headless.
"""
import os

from datasets import Dataset, load_from_disk

from local_save import local_save_path, save_dataset_locally


def test_local_save_path_is_root_task_split():
    assert local_save_path("datasets/slapo_local", "push-v3", "train") == os.path.join(
        "datasets/slapo_local", "push-v3", "train"
    )


def test_save_dataset_locally_writes_reloadable_layout(tmp_path):
    ds = Dataset.from_dict({"observation": [1, 2, 3], "action": [[0.0], [1.0], [2.0]]})
    root = str(tmp_path / "slapo_local")

    path = save_dataset_locally(ds, root, "push-v3", "train")

    assert path == os.path.join(root, "push-v3", "train")
    assert os.path.isdir(path)
    reloaded = load_from_disk(path)
    assert len(reloaded) == 3
    assert reloaded[0]["observation"] == 1
    assert reloaded[2]["action"] == [2.0]


def test_save_dataset_locally_overwrites_previous_copy(tmp_path):
    root = str(tmp_path / "slapo_local")
    save_dataset_locally(Dataset.from_dict({"observation": [1, 2, 3]}), root, "push-v3", "train")
    # A fresh regeneration with different content must replace, not merge/duplicate.
    path = save_dataset_locally(Dataset.from_dict({"observation": [9, 9]}), root, "push-v3", "train")
    reloaded = load_from_disk(path)
    assert len(reloaded) == 2
    assert reloaded[0]["observation"] == 9
