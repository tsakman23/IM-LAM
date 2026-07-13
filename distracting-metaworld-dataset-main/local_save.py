"""Persist a generated dataset locally in a training-loadable layout.

Isolated from generate_dataset_huggingface.py (no metaworld/mujoco imports) so it can be
unit-tested fast and headless. The layout <root>/<task>/<split> is a `save_to_disk` directory
per (config, split); the training loader (`ifo.common.data.huggingface.HuggingFaceDataset`)
reads it via `load_from_disk` when `dataset.dataset_path` points at <root>, skipping the
generate -> push-to-HF -> download-back round-trip.
"""
import os
import shutil

from datasets import Dataset


def local_save_path(root: str, task: str, split: str) -> str:
    """Return the <root>/<task>/<split> directory for a (task, split) local dataset."""
    return os.path.join(root, task, split)


def save_dataset_locally(dataset: Dataset, root: str, task: str, split: str) -> str:
    """Save ``dataset`` to <root>/<task>/<split> via ``save_to_disk`` and return that path.

    A regeneration produces fresh data, so an existing copy at that path is replaced (not
    duplicated) - this keeps re-runs from leaving stale/partial copies behind.
    """
    path = local_save_path(root, task, split)
    if os.path.isdir(path):
        shutil.rmtree(path)
    dataset.save_to_disk(path)
    return path
