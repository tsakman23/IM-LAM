"""Tests for local (save_to_disk / load_from_disk) dataset loading in HuggingFaceDataset.

The local branch lets training read a generator-produced dataset directly from disk,
skipping the HF download round-trip. A non-local ``path`` (a repo id) must still route
to ``load_dataset`` unchanged.
"""
import os

from datasets import Dataset

from ifo.common.data.huggingface import HuggingFaceDataset
from ifo.common.utils.data import get_dataset


def _make_local_dataset(root, config, split, n=5):
    """Write a tiny save_to_disk dataset at <root>/<config>/<split> and return n."""
    data = Dataset.from_dict(
        {
            "observation": [[float(i)] for i in range(n)],
            "action": [[float(i), float(-i)] for i in range(n)],
        }
    )
    data.save_to_disk(os.path.join(root, config, split))
    return n


def test_loads_from_local_dir_via_load_from_disk(tmp_path):
    root = str(tmp_path / "slapo_local")
    n = _make_local_dataset(root, "push-v3", "train")

    ds = HuggingFaceDataset(
        path=root, name="push-v3", split="train",
        columns=["observation", "action"], transform=None, block_size=1,
    )

    assert len(ds) == n
    sample = ds[0]
    assert set(sample.keys()) == {"observation", "action"}


def test_repo_id_path_still_uses_load_dataset(tmp_path, monkeypatch):
    # A non-local path (repo id) must NOT hit the local branch: it should call load_dataset.
    calls = {}

    def fake_load_dataset(path, name=None, split=None, **kwargs):
        calls["path"] = path
        calls["name"] = name
        return Dataset.from_dict({"observation": [[1.0]], "action": [[0.0, 0.0]]})

    def fail_load_from_disk(*a, **k):
        raise AssertionError("load_from_disk must not be called for a repo id")

    monkeypatch.setattr("ifo.common.data.huggingface.load_dataset", fake_load_dataset)
    monkeypatch.setattr("ifo.common.data.huggingface.load_from_disk", fail_load_from_disk, raising=False)

    HuggingFaceDataset(
        path="tsakman23/visual_masked_distracting_metaworld", name="push-v3", split="train",
        columns=["observation", "action"], transform=None, block_size=1,
    )
    assert calls["path"] == "tsakman23/visual_masked_distracting_metaworld"
    assert calls["name"] == "push-v3"


def test_get_dataset_reads_local_metaworld_object_mask_dataset(tmp_path):
    # Full training consumption path over a local (save_to_disk) object-mask dataset:
    # get_dataset -> MetaWorldDataset (rename observation_distracted->observation, object_mask,
    # column select) -> HuggingFaceDataset local branch. No network, no env.
    n = 4
    img = [[[0, 0, 0], [255, 255, 255]], [[10, 20, 30], [40, 50, 60]]]  # (2,2,3)
    m = [[0, 255], [255, 0]]  # (2,2)
    data = Dataset.from_dict(
        {
            "observation": [img] * n,
            "observation_distracted": [img] * n,
            "mask": [m] * n,
            "object_mask": [m] * n,
            "action": [[0.1, -0.2, 0.3, -0.4]] * n,
        }
    )
    root = str(tmp_path / "slapo_local")
    data.save_to_disk(os.path.join(root, "push-v3", "test"))

    ds = get_dataset(
        name="Meta-World/masked-MT1-push-v3", split="test", cache_dir=str(tmp_path / "cache"),
        block_size=1, dataset_path=root, with_object_mask=True,
    )

    assert len(ds) == n
    sample = ds[0]
    assert {"observation", "mask", "object_mask", "action"} <= set(sample.keys())
    # ProcessMask binarizes to {0,1}; center maps uint8 pixels to [-0.5, 0.5].
    assert set(sample["object_mask"].unique().tolist()) <= {0.0, 1.0}
    assert float(sample["observation"].min()) >= -0.5 and float(sample["observation"].max()) <= 0.5
