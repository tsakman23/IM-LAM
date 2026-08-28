"""GPU integration test for the SAM2 dual-object predictor. Run with:
    pytest tests/test_episode_integration.py -m integration
"""
import numpy as np
import pytest
import torch
from PIL import Image

SAM2_ID = "facebook/sam2-hiera-tiny"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _frames(n=5, size=64):
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8), "RGB")
        for _ in range(n)
    ]


@pytest.mark.integration
def test_process_episode_returns_aligned_binary_masks_for_both_entities():
    from episode import Sam2DualPredictor

    frames = _frames(5, 64)
    predictor = Sam2DualPredictor(SAM2_ID, "cuda:0", torch.bfloat16)
    out = predictor.process_episode(
        frames, agent_box=[8, 8, 56, 56], object_box=[20, 20, 34, 34]
    )
    assert set(out) == {"agent", "object"}
    for key in ("agent", "object"):
        assert len(out[key]) == len(frames)
        for m in out[key]:
            assert isinstance(m, Image.Image)
            assert m.mode == "L"
            assert m.size == frames[0].size
            vals = set(np.unique(np.asarray(m)).tolist())
            assert vals.issubset({0, 255})


@pytest.mark.integration
def test_process_episode_without_object_box_yields_empty_object_masks():
    from episode import Sam2DualPredictor

    frames = _frames(4, 64)
    predictor = Sam2DualPredictor(SAM2_ID, "cuda:0", torch.bfloat16)
    out = predictor.process_episode(frames, agent_box=[8, 8, 56, 56], object_box=None)
    assert len(out["object"]) == len(frames)
    assert all(np.asarray(m).sum() == 0 for m in out["object"])
    # Agent should still be tracked (non-trivial mask on at least one frame).
    assert any(np.asarray(m).sum() > 0 for m in out["agent"])
