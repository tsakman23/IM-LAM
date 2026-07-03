import os
from typing import List, Optional

import h5py
import numpy as np
import torch
from pypdl import Pypdl
from tensordict import TensorDict
from torch.utils.data import Dataset

from ifo.common.transforms.data import DictTransform

DISTRACTING_CONTROL_SUITE_DATASET_URLS = {
    # cheetah-run-distracting
    "cheetah-run-scale-easy-video-hard-64px-5k": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-scale-easy-video-hard-64px-5k.hdf5",
    "cheetah-run-scale-easy-video-hard-64px-eval": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-scale-easy-video-hard-64px-eval.hdf5",
    "cheetah-run-scale-easy-video-hard-labeled-1000x128-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-scale-easy-video-hard-labeled-1000x128-64px.hdf5",
    "cheetah-run-scale-easy-video-hard-labeled-1000x64-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-scale-easy-video-hard-labeled-1000x64-64px.hdf5",
    "cheetah-run-scale-easy-video-hard-labeled-1000x32-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-scale-easy-video-hard-labeled-1000x32-64px.hdf5",
    "cheetah-run-scale-easy-video-hard-labeled-1000x16-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-scale-easy-video-hard-labeled-1000x16-64px.hdf5",
    "cheetah-run-scale-easy-video-hard-labeled-1000x8-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-scale-easy-video-hard-labeled-1000x8-64px.hdf5",
    "cheetah-run-scale-easy-video-hard-labeled-1000x4-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-scale-easy-video-hard-labeled-1000x4-64px.hdf5",
    "cheetah-run-scale-easy-video-hard-labeled-1000x2-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-scale-easy-video-hard-labeled-1000x2-64px.hdf5",
    # cheetah-run-vanilla
    "cheetah-run-vanilla-64px-5k": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-vanilla-64px-5k.hdf5",
    "cheetah-run-vanilla-labeled-1000x128-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-vanilla-labeled-1000x128-64px.hdf5",
    "cheetah-run-vanilla-labeled-1000x64-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-vanilla-labeled-1000x64-64px.hdf5",
    "cheetah-run-vanilla-labeled-1000x32-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-vanilla-labeled-1000x32-64px.hdf5",
    "cheetah-run-vanilla-labeled-1000x16-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-vanilla-labeled-1000x16-64px.hdf5",
    "cheetah-run-vanilla-labeled-1000x8-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-vanilla-labeled-1000x8-64px.hdf5",
    "cheetah-run-vanilla-labeled-1000x4-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-vanilla-labeled-1000x4-64px.hdf5",
    "cheetah-run-vanilla-labeled-1000x2-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/cheetah-run-vanilla-labeled-1000x2-64px.hdf5",
    # walker-run-distracting
    "walker-run-scale-easy-video-hard-64px-5k": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-scale-easy-video-hard-64px-5k.hdf5",
    "walker-run-scale-easy-video-hard-64px-eval": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-scale-easy-video-hard-64px-eval.hdf5",
    "walker-run-scale-easy-video-hard-labeled-1000x128-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-scale-easy-video-hard-labeled-1000x128-64px.hdf5",
    "walker-run-scale-easy-video-hard-labeled-1000x64-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-scale-easy-video-hard-labeled-1000x64-64px.hdf5",
    "walker-run-scale-easy-video-hard-labeled-1000x32-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-scale-easy-video-hard-labeled-1000x32-64px.hdf5",
    "walker-run-scale-easy-video-hard-labeled-1000x16-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-scale-easy-video-hard-labeled-1000x16-64px.hdf5",
    "walker-run-scale-easy-video-hard-labeled-1000x8-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-scale-easy-video-hard-labeled-1000x8-64px.hdf5",
    "walker-run-scale-easy-video-hard-labeled-1000x4-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-scale-easy-video-hard-labeled-1000x4-64px.hdf5",
    "walker-run-scale-easy-video-hard-labeled-1000x2-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-scale-easy-video-hard-labeled-1000x2-64px.hdf5",
    # walker-run-vanilla
    "walker-run-vanilla-64px-5k": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-vanilla-64px-5k.hdf5",
    "walker-run-vanilla-labeled-1000x128-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-vanilla-labeled-1000x128-64px.hdf5",
    "walker-run-vanilla-labeled-1000x64-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-vanilla-labeled-1000x64-64px.hdf5",
    "walker-run-vanilla-labeled-1000x32-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-vanilla-labeled-1000x32-64px.hdf5",
    "walker-run-vanilla-labeled-1000x16-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-vanilla-labeled-1000x16-64px.hdf5",
    "walker-run-vanilla-labeled-1000x8-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-vanilla-labeled-1000x8-64px.hdf5",
    "walker-run-vanilla-labeled-1000x4-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-vanilla-labeled-1000x4-64px.hdf5",
    "walker-run-vanilla-labeled-1000x2-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/walker-run-vanilla-labeled-1000x2-64px.hdf5",
    # hopper-hop
    "hopper-hop-scale-easy-video-hard-64px-5k": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-scale-easy-video-hard-64px-5k.hdf5",
    "hopper-hop-scale-easy-video-hard-64px-eval": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-scale-easy-video-hard-64px-eval.hdf5",
    "hopper-hop-scale-easy-video-hard-labeled-1000x128-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-scale-easy-video-hard-labeled-1000x128-64px.hdf5",
    "hopper-hop-scale-easy-video-hard-labeled-1000x64-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-scale-easy-video-hard-labeled-1000x64-64px.hdf5",
    "hopper-hop-scale-easy-video-hard-labeled-1000x32-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-scale-easy-video-hard-labeled-1000x32-64px.hdf5",
    "hopper-hop-scale-easy-video-hard-labeled-1000x16-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-scale-easy-video-hard-labeled-1000x16-64px.hdf5",
    "hopper-hop-scale-easy-video-hard-labeled-1000x8-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-scale-easy-video-hard-labeled-1000x8-64px.hdf5",
    "hopper-hop-scale-easy-video-hard-labeled-1000x4-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-scale-easy-video-hard-labeled-1000x4-64px.hdf5",
    "hopper-hop-scale-easy-video-hard-labeled-1000x2-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-scale-easy-video-hard-labeled-1000x2-64px.hdf5",
    # hopper-hop-vanilla
    "hopper-hop-vanilla-64px-5k": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-vanilla-64px-5k.hdf5",
    "hopper-hop-vanilla-labeled-1000x128-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-vanilla-labeled-1000x128-64px.hdf5",
    "hopper-hop-vanilla-labeled-1000x64-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-vanilla-labeled-1000x64-64px.hdf5",
    "hopper-hop-vanilla-labeled-1000x32-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-vanilla-labeled-1000x32-64px.hdf5",
    "hopper-hop-vanilla-labeled-1000x16-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-vanilla-labeled-1000x16-64px.hdf5",
    "hopper-hop-vanilla-labeled-1000x8-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-vanilla-labeled-1000x8-64px.hdf5",
    "hopper-hop-vanilla-labeled-1000x4-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-vanilla-labeled-1000x4-64px.hdf5",
    "hopper-hop-vanilla-labeled-1000x2-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/hopper-hop-vanilla-labeled-1000x2-64px.hdf5",
    # humanoid-walk-distracting
    "humanoid-walk-scale-easy-video-hard-64px-5k": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-scale-easy-video-hard-64px-5k.hdf5",
    "humanoid-walk-scale-easy-video-hard-64px-eval": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-scale-easy-video-hard-64px-eval.hdf5",
    "humanoid-walk-scale-easy-video-hard-labeled-1000x128-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-scale-easy-video-hard-labeled-1000x128-64px.hdf5",
    "humanoid-walk-scale-easy-video-hard-labeled-1000x64-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-scale-easy-video-hard-labeled-1000x64-64px.hdf5",
    "humanoid-walk-scale-easy-video-hard-labeled-1000x32-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-scale-easy-video-hard-labeled-1000x32-64px.hdf5",
    "humanoid-walk-scale-easy-video-hard-labeled-1000x16-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-scale-easy-video-hard-labeled-1000x16-64px.hdf5",
    "humanoid-walk-scale-easy-video-hard-labeled-1000x8-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-scale-easy-video-hard-labeled-1000x8-64px.hdf5",
    "humanoid-walk-scale-easy-video-hard-labeled-1000x4-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-scale-easy-video-hard-labeled-1000x4-64px.hdf5",
    "humanoid-walk-scale-easy-video-hard-labeled-1000x2-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-scale-easy-video-hard-labeled-1000x2-64px.hdf5",
    # humanoid-walk-vanilla
    "humanoid-walk-vanilla-64px-5k": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-vanilla-64px-5k.hdf5",
    "humanoid-walk-vanilla-labeled-1000x128-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-vanilla-labeled-1000x128-64px.hdf5",
    "humanoid-walk-vanilla-labeled-1000x64-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-vanilla-labeled-1000x64-64px.hdf5",
    "humanoid-walk-vanilla-labeled-1000x32-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-vanilla-labeled-1000x32-64px.hdf5",
    "humanoid-walk-vanilla-labeled-1000x16-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-vanilla-labeled-1000x16-64px.hdf5",
    "humanoid-walk-vanilla-labeled-1000x8-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-vanilla-labeled-1000x8-64px.hdf5",
    "humanoid-walk-vanilla-labeled-1000x4-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-vanilla-labeled-1000x4-64px.hdf5",
    "humanoid-walk-vanilla-labeled-1000x2-64px": "https://fors3toevodomain.s3.cloud.ru/lapo-data/humanoid-walk-vanilla-labeled-1000x2-64px.hdf5",
}


class DistractingControlSuiteDataset(Dataset):
    """
    A dataset containing expert trajectories from the Distracting Control Suite.

    This dataset provides access to expert trajectories from four environments
    (cheetah, walker, hopper, humanoid) with various distractions. The data is
    provided as HDF5 files, which are downloaded on first use.

    Each data instance represents a single step consisting of tuples of the form:
    (observation, action, state)

    Available dataset names can be inspected from `DISTRACTING_CONTROL_SUITE_DATASET_URLS.keys()`.

    Returns:
        Dataset: A dataset containing expert trajectories from the Distracting Control Suite.

    The average returns for each environment are:
    - cheetah-run: 837.70
    - walker-run: 739.79
    - hopper-hop: 306.63
    - humanoid-walk: 617.22

    This dataset is related to the following papers:
    - Distracting Control Suite: https://arxiv.org/abs/2101.02722
    - Latent Action Learning Requires Supervision in the Presence of Distractors: https://github.com/dunnolab/laom

    Data Fields:
    - observation: The current RGB observation from the environment.
    - state: The current state of the environment.
    - action: The action taken by the expert policy.
    """

    _renamed_columns = {
        "actions": "action",
        "obs": "observation",
        "states": "state",
    }
    _renamed_columns_inv = {v: k for k, v in _renamed_columns.items()}

    def __init__(
        self,
        name: str,
        cache_dir: Optional[str] = None,
        block_size: Optional[int] = None,
        columns: Optional[List[str]] = None,
        transform: Optional[DictTransform] = None,
        **kwargs,
    ) -> None:
        """
        Initialize DistractingControlSuiteDataset.

        Args:
            name (str): Name of the dataset configuration.
            cache_dir (str, optional): Directory to read/write data.
                Defaults to ~/.cache/ifo/distracting_control_suite.
            block_size (int, optional): Block size of the data. Data gets an
                additional temporal dimension. Defaults to 1.
            columns (List[str], optional): Columns to load. If None, all columns are loaded.
            transform (DictTransform, optional): Transform to apply.
        """
        assert (
            name in DISTRACTING_CONTROL_SUITE_DATASET_URLS
        ), f"Dataset '{name}' not found. Available datasets: {list(DISTRACTING_CONTROL_SUITE_DATASET_URLS.keys())}"
        assert isinstance(transform, (DictTransform, type(None))), "Transforms must be of type DictTransform!"

        self.block_size = block_size if block_size else 1
        self.transform = transform
        self.columns = columns

        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/ifo/distracting_control_suite")

        url = DISTRACTING_CONTROL_SUITE_DATASET_URLS[name]
        filename = os.path.basename(url)
        filepath = os.path.join(cache_dir, filename)
        self._filepath = filepath

        if not os.path.exists(filepath):
            print(f"Downloading dataset {name} to {filepath}...")
            os.makedirs(cache_dir, exist_ok=True)
            dl = Pypdl()
            dl.start(url=url, file_path=filepath)

        with h5py.File(filepath, "r") as data:
            if self.columns is None:
                self.columns = list(data["0"].keys())
                self.columns = [self._renamed_columns[c] for c in self.columns]

            assert all(
                [c in self._renamed_columns_inv.keys() for c in self.columns]
            ), f"Columns {self.columns} not found in dataset"

            self._columns = [self._renamed_columns_inv[c] for c in self.columns]
            self.traj_len = data["0"][self._columns[0]].shape[0]
            self.traj_count = len(data.keys())
            self.length = self.traj_count * (self.traj_len - self.block_size + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> TensorDict:
        """
        Get the item at the specified index and return a copy.

        Args:
            index (int): The index of the item to get.

        Returns:
            TensorDict: A dictionary containing the item's data.
        """
        assert isinstance(index, int), "Index must be an integer"
        traj_idx, trans_idx = divmod(index, self.traj_len - self.block_size + 1)
        with h5py.File(self._filepath, "r") as data:
            sample = {
                self._renamed_columns[c]: torch.from_numpy(
                    data[f"{traj_idx}"][c][trans_idx : trans_idx + self.block_size]
                    )
                for c in self._columns
            }
        sample = TensorDict(sample, batch_size=(self.block_size,))
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __getitems__(self, indices: List[int]) -> TensorDict:
        """Get a batch of items from the dataset.

        Args:
            indices (List[int]): The indices of the items to get.

        Returns:
            TensorDict: A dictionary containing the items' data.
        """
        assert isinstance(indices, list) and isinstance(indices[0], int), "Indices must be a list of integers"
        traj_indices, trans_indices = np.divmod(indices, self.traj_len - self.block_size + 1)
        samples = {k: [] for k in self.columns}
        with h5py.File(self._filepath, "r") as data:
            for traj_idx, trans_idx in zip(traj_indices, trans_indices):
                for k in self.columns:
                    samples[k].append(
                        torch.from_numpy(
                            data[f"{traj_idx}"][self._renamed_columns_inv[k]][trans_idx : trans_idx + self.block_size]
                            )
                        )
        samples = TensorDict(samples, batch_size=(len(indices), self.block_size))
        if self.transform:
            samples = self.transform(samples)
        return samples
