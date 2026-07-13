"""Tests for the reusable path-B write helpers in scripts/save_local_dataset.py.

Idempotency contract: skip when a completed local copy already exists (so re-running
generation/download never duplicates or stubs), overwrite when explicitly told to.
Uses only `datasets` (no torch), so it stays light.
"""
import os
import sys

from datasets import Dataset, load_from_disk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from save_local_dataset import local_dest, write_local_copy  # noqa: E402


def _ds(n):
    return Dataset.from_dict({"observation": [[i] for i in range(n)], "action": [[0.0]] * n})


def test_local_dest_layout():
    assert local_dest("root", "push-v3", "train") == os.path.join("root", "push-v3", "train")


def test_write_creates_loadable_copy(tmp_path):
    dest = str(tmp_path / "push-v3" / "train")
    wrote = write_local_copy(_ds(3), dest, skip_if_exists=True)
    assert wrote is True
    assert len(load_from_disk(dest)) == 3


def test_write_skips_when_completed_copy_present(tmp_path):
    dest = str(tmp_path / "push-v3" / "train")
    write_local_copy(_ds(3), dest, skip_if_exists=True)
    wrote = write_local_copy(_ds(5), dest, skip_if_exists=True)  # different data, should be ignored
    assert wrote is False
    assert len(load_from_disk(dest)) == 3  # unchanged


def test_write_overwrites_when_not_skipping(tmp_path):
    dest = str(tmp_path / "push-v3" / "train")
    write_local_copy(_ds(3), dest, skip_if_exists=False)
    write_local_copy(_ds(5), dest, skip_if_exists=False)  # fresh generation overwrites
    assert len(load_from_disk(dest)) == 5
