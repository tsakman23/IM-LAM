"""GPU+data integration test for run_worker on real cached push-v3 parquet.
    pytest tests/test_worker_integration.py -m integration
"""
import glob
import json
import os

import numpy as np
import pytest
import torch
from datasets import Dataset

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_GLOB = ("/data2/masklam/datasets/hf_home/hub/"
             "datasets--tsakman23--visual_masked_distracting_metaworld/snapshots/*/"
             "push-v3/train-00000-of-*.parquet")


@pytest.mark.integration
def test_run_worker_produces_superset_shard_on_real_data(tmp_path):
    from worker import run_worker

    src = glob.glob(_SRC_GLOB)
    if not src:
        pytest.skip("cached push-v3 parquet not found")

    with open(os.path.join(_PKG, "config.json")) as f:
        config = json.load(f)
    config["output_dir"] = str(tmp_path)
    config["grounding_dino_model_id"] = "IDEA-Research/grounding-dino-tiny"  # speed

    stats = run_worker("push-v3", "train", config, device="cuda:0",
                       source=src, max_episodes=2)

    assert stats["episodes"] == 2
    assert 0.0 <= stats["detection_rate"] <= 1.0
    assert stats["frames"] > 0

    shards = glob.glob(os.path.join(tmp_path, "push-v3", "train", "*.parquet"))
    assert shards, "no output shard written"
    out = Dataset.from_parquet(shards)
    assert "pred_mask" in out.column_names
    assert "pred_object_mask" in out.column_names
    assert len(out) == stats["frames"]

    # No-regeneration invariant on real data: first output frame's observation
    # is pixel-identical to the source's first frame.
    src_ds = Dataset.from_parquet(src)
    np.testing.assert_array_equal(
        np.asarray(out[0]["observation"]), np.asarray(src_ds[0]["observation"])
    )
    # pred masks are single-channel and frame-sized.
    pm = out[0]["pred_mask"]
    assert pm.mode == "L"
    assert pm.size == src_ds[0]["observation"].size
