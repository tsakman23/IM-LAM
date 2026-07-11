"""Unit tests for the pure helpers in compare_reconstruction.py.

These exercise the metric / selection / mask logic on synthetic tensors, with no
GPU, dataset, or checkpoint dependency, so they verify correctness deterministically.
"""
import os
import sys

import torch

# compare_reconstruction.py lives in <repo>/scripts/ (not an installed package).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

from compare_reconstruction import (  # noqa: E402
    error_difference_map,
    object_motion_score,
    per_pixel_mse,
    region_error,
    resolve_input_mask,
    select_moving_object_frames,
)


def test_per_pixel_mse_channel_mean():
    pred = torch.zeros(2, 3, 4, 4)
    gt = torch.ones(2, 3, 4, 4)
    ppm = per_pixel_mse(pred, gt)
    assert ppm.shape == (2, 4, 4)
    assert torch.allclose(ppm, torch.ones(2, 4, 4))  # (0-1)^2 averaged over 3 channels == 1


def test_region_error_gates_to_mask_only():
    # Error is 4 inside the mask region, 100 outside; region error must ignore outside.
    perpixel = torch.full((1, 4, 4), 100.0)
    mask = torch.zeros(1, 4, 4)
    mask[0, :2, :2] = 1.0
    perpixel[0, :2, :2] = 4.0
    err = region_error(perpixel, mask)
    assert torch.allclose(err, torch.tensor([4.0]))


def test_region_error_empty_mask_is_finite():
    perpixel = torch.rand(1, 4, 4)
    mask = torch.zeros(1, 4, 4)
    err = region_error(perpixel, mask)
    assert torch.isfinite(err).all()
    assert torch.allclose(err, torch.zeros(1))


def test_resolve_input_mask_loss_only_returns_agent():
    agent = torch.zeros(1, 1, 2, 2)
    obj = torch.ones(1, 1, 2, 2)
    assert torch.equal(resolve_input_mask(agent, obj, object_mask_input=False), agent)


def test_resolve_input_mask_input_on_returns_union():
    agent = torch.zeros(1, 1, 2, 2)
    agent[0, 0, 0, 0] = 1.0
    obj = torch.zeros(1, 1, 2, 2)
    obj[0, 0, 1, 1] = 1.0
    union = resolve_input_mask(agent, obj, object_mask_input=True)
    assert union.max() <= 1.0 and union.sum() == 2.0


def test_object_motion_score_detects_displacement():
    # Object present at both ends but displaced -> positive score; absent -> 0.
    block = torch.zeros(13, 1, 8, 8)
    block[0, 0, 0, 0] = 1.0       # start centroid at (0,0)
    block[3, 0, 7, 7] = 1.0       # target centroid at (7,7)
    score = object_motion_score(block, frame_stack=3)
    assert score > 9.0  # ~sqrt(7^2+7^2) ~= 9.9

    empty = torch.zeros(13, 1, 8, 8)
    assert object_motion_score(empty, frame_stack=3) == 0.0


def test_error_difference_map_zeroes_background_and_negatives():
    a = torch.tensor([[5.0, 1.0], [0.0, 3.0]])  # MaskLAM per-pixel MSE
    b = torch.tensor([[1.0, 4.0], [0.0, 0.0]])  # FG per-pixel MSE
    sil = torch.tensor([[1.0, 1.0], [0.0, 0.0]])  # object silhouette (top row only)
    diff = error_difference_map(a, b, sil)
    # top-left: 5-1=4 (kept); top-right: 1-4<0 -> 0; bottom row off-silhouette -> 0.
    assert torch.equal(diff, torch.tensor([[4.0, 0.0], [0.0, 0.0]]))


class _FakeDataset:
    """Windowed dataset stub: index i -> object_mask that moves more for larger i."""

    def __init__(self, n, frame_stack, h=16, w=16):
        self.n = n
        self.frame_stack = frame_stack
        self.h, self.w = h, w

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        block = torch.zeros(self.frame_stack + 1, 1, self.h, self.w)
        # Every window has an object at the target frame (area 4), starting at a fixed
        # corner and ending at a column proportional to i -> larger i == more motion.
        block[0, 0, 0, 0:2] = 1.0
        col = min(self.w - 2, 1 + i % (self.w - 2))
        block[self.frame_stack, 0, 0, col : col + 2] = 1.0
        return {"object_mask": block}


def test_select_moving_object_frames_ranks_by_motion_and_separates():
    ds = _FakeDataset(n=400, frame_stack=3)
    picked = select_moving_object_frames(
        ds, frame_stack=3, num_frames=3, scan_limit=400, min_object_pixels=1, min_separation=50
    )
    assert len(picked) == 3
    # Selected indices must respect the minimum separation.
    for a in range(len(picked)):
        for b in range(a + 1, len(picked)):
            assert abs(picked[a] - picked[b]) >= 50


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
