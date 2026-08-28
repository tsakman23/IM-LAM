"""Round-trip invariant: after the extractor writes a superset shard, every
original DMW column must be recoverable unchanged - images pixel-identical,
scalars/sequences exact. This enforces the no-regeneration guarantee: the
extractor only appends pred_mask/pred_object_mask; it never alters the frames
or any other original column. No GPU (fake predictor), no network.
"""
import numpy as np
from datasets import Dataset, Features, Sequence, Value
from datasets import Image as HFImage
from PIL import Image

from worker import extract_episode, write_shard

_IMAGE_COLS = ["observation", "observation_distracted", "mask", "object_mask"]
_ORIG_COLS = _IMAGE_COLS + ["state", "action", "reward", "terminated", "truncated"]


class _FakePredictor:
    def process_episode(self, frames, agent_box, object_box):
        n = len(frames)
        w, h = frames[0].size
        return {
            "agent": [Image.new("L", (w, h), 255) for _ in range(n)],
            "object": [Image.new("L", (w, h), 128) for _ in range(n)],
        }


class _FakeDetector:
    def detect_object(self, frame, prompt, *a, **k):
        return [1, 2, 3, 4], True


def _img(seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (6, 6, 3), dtype=np.uint8), "RGB")


def _build_source(path):
    n = 3
    rows = {
        "observation": [_img(10 + i) for i in range(n)],
        "observation_distracted": [_img(20 + i) for i in range(n)],
        "mask": [_img(30 + i) for i in range(n)],
        "object_mask": [_img(40 + i) for i in range(n)],
        "state": [[float(i), float(i) + 0.5] for i in range(n)],
        "action": [[0.1 * i, -0.1 * i] for i in range(n)],
        "reward": [float(i) * 0.25 for i in range(n)],
        "terminated": [False, False, True],
        "truncated": [False, False, False],
    }
    feats = Features({
        "observation": HFImage(), "observation_distracted": HFImage(),
        "mask": HFImage(), "object_mask": HFImage(),
        "state": Sequence(Value("float32")), "action": Sequence(Value("float32")),
        "reward": Value("float32"), "terminated": Value("bool"),
        "truncated": Value("bool"),
    })
    Dataset.from_dict(rows, features=feats).to_parquet(str(path))
    return rows


def test_shard_roundtrip_preserves_original_columns(tmp_path):
    src_path = tmp_path / "source.parquet"
    src_rows = _build_source(src_path)

    # Stream the source back exactly as the worker would, then extract.
    loaded = Dataset.from_parquet(str(src_path))
    episode_rows = list(loaded)
    out, meta = extract_episode(
        episode_rows, _FakeDetector(), _FakePredictor(),
        cfg={"object_prompt": "a puck.", "agent_bbox": [0, 0, 5, 5],
             "box_threshold": 0.3, "text_threshold": 0.25,
             "fallback_box_threshold": 0.15, "fallback_text_threshold": 0.15},
        obs_col="observation_distracted", orig_columns=_ORIG_COLS,
    )

    out_path = tmp_path / "shard.parquet"
    write_shard(out, str(out_path), image_columns=_IMAGE_COLS + ["pred_mask", "pred_object_mask"])
    result = Dataset.from_parquet(str(out_path))

    assert len(result) == 3
    # New columns exist.
    assert "pred_mask" in result.column_names
    assert "pred_object_mask" in result.column_names

    for i, row in enumerate(result):
        # Images: pixel-identical to the source.
        for col in _IMAGE_COLS:
            np.testing.assert_array_equal(
                np.asarray(row[col]), np.asarray(src_rows[col][i]),
                err_msg=f"{col} row {i} pixels changed",
            )
        # Scalars / sequences: exact.
        np.testing.assert_allclose(row["state"], src_rows["state"][i])
        np.testing.assert_allclose(row["action"], src_rows["action"][i])
        assert row["reward"] == src_rows["reward"][i]
        assert row["terminated"] == src_rows["terminated"][i]
        assert row["truncated"] == src_rows["truncated"][i]
