from typing import List, Optional

import torch
from datasets import load_dataset
from tensordict import TensorDict
from torch.utils.data import Dataset

from src.data.transforms import DictTransform


def collate_fn(batch):
    """Identity collate — __getitems__ already returns a batched TensorDict."""
    return batch


class HuggingFaceDataset(Dataset):
    """Base dataset wrapping HuggingFace datasets with temporal windowing support."""

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        split: str = "train",
        cache_dir: Optional[str] = None,
        num_proc: Optional[int] = None,
        block_size: Optional[int] = None,
        columns: Optional[List[str]] = None,
        transform: Optional[DictTransform] = None,
        **kwargs,
    ) -> None:
        assert isinstance(transform, (DictTransform, type(None))), "Transforms must be of type DictTransform!"
        self.block_size = block_size if block_size else 0
        self.transform = transform
        self.columns = columns
        self.data = load_dataset(path=path, name=name, cache_dir=cache_dir, split=split, num_proc=num_proc, **kwargs)
        self.data = self.data.with_format("torch", columns=columns)

    def __len__(self) -> int:
        if self.block_size:
            return len(self.data) - self.block_size + 1
        else:
            return len(self.data)

    def __getitems__(self, indices: List[int]) -> TensorDict:
        """Get a batch of items, optionally with temporal windowing."""
        assert isinstance(indices, list) and isinstance(indices[0], int)
        if self.block_size:
            indices = torch.as_tensor(indices)[:, None] + torch.arange(self.block_size)[None, :]
            samples = self.data[indices.flatten()]
            samples = TensorDict(samples, batch_size=(len(indices) * self.block_size,))
            samples = samples.view(indices.shape[0], self.block_size)
        else:
            samples = self.data[indices]
            samples = TensorDict(samples, batch_size=(len(indices),))
        if self.transform:
            samples = self.transform(samples)
        return samples

    def __getitem__(self, index: int) -> TensorDict:
        """Get a single item with optional temporal windowing."""
        assert isinstance(index, int)
        sample = TensorDict(self.data[index : index + self.block_size], batch_size=(self.block_size,))
        if self.transform:
            sample = self.transform(sample)
        return sample


class MaskedDistractingControlSuiteDataset(HuggingFaceDataset):
    """Dataset for Distracting Control Suite with segmentation masks.

    HF path: EpicPinkPenguin/visual_distracting_control_suite
    Configs: cheetah_run, cheetah_run_distractor_low, cheetah_run_distractor_hard,
             walker_run*, hopper_hop*, humanoid_walk*
    """

    def __init__(
        self,
        name: Optional[str] = None,
        split: str = "train",
        cache_dir: Optional[str] = None,
        num_proc: Optional[int] = None,
        block_size: Optional[int] = None,
        columns: Optional[List[str]] = None,
        transform: Optional[DictTransform] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            path="EpicPinkPenguin/visual_distracting_control_suite",
            name=name,
            split=split,
            cache_dir=cache_dir,
            num_proc=num_proc,
            block_size=block_size,
            columns=columns,
            transform=transform,
            **kwargs,
        )
        assert "mask" in self.data.column_names, (
            f"Dataset requires a 'mask' column, but got {self.data.column_names}"
        )


class MetaWorldDataset(HuggingFaceDataset):
    """Dataset for Meta-World with optional distracted observations.

    HF path: EpicPinkPenguin/visual_distracting_metaworld
    Configs: hammer-v3, bin-picking-v3, basketball-v3, soccer-v3
    """

    def __init__(
        self,
        name: Optional[str] = None,
        split: str = "train",
        cache_dir: Optional[str] = None,
        num_proc: Optional[int] = None,
        block_size: Optional[int] = None,
        columns: Optional[List[str]] = None,
        transform: Optional[DictTransform] = None,
        use_distracted_obs: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            path="EpicPinkPenguin/visual_distracting_metaworld",
            name=name,
            split=split,
            cache_dir=cache_dir,
            num_proc=num_proc,
            block_size=block_size,
            columns=columns,
            transform=transform,
            **kwargs,
        )
        assert "mask" in self.data.column_names, (
            f"Dataset requires a 'mask' column, but got {self.data.column_names}"
        )
        if use_distracted_obs and "observation_distracted" in self.data.column_names:
            self.data = self.data.remove_columns("observation")
            self.data = self.data.rename_column("observation_distracted", "observation")
            self.data = self.data.with_format("torch", columns=self.columns)
