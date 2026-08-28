"""Unit tests for worker orchestration logic - no GPU, no network. The detector
and predictor are injected as fakes so episode grouping and the verbatim
column pass-through (the no-regeneration invariant) are tested in isolation.
"""
import numpy as np
from PIL import Image

from worker import extract_episode, iter_episodes

_CFG = {
    "object_prompt": "a puck.",
    "agent_bbox": [1, 2, 3, 4],
    "box_threshold": 0.3, "text_threshold": 0.25,
    "fallback_box_threshold": 0.15, "fallback_text_threshold": 0.15,
}


class _FakeDetector:
    def __init__(self, box=(5, 6, 7, 8), detected=True):
        self.box, self.detected = box, detected

    def detect_object(self, frame, prompt, box_threshold, text_threshold,
                      fallback_box_threshold, fallback_text_threshold):
        return (list(self.box) if self.detected else None, self.detected)


class _FakePredictor:
    """Agent mask = all 255; object mask = all 200 if an object box was passed,
    else all 0. Lets tests assert both the pass-through and the None-box path."""

    def process_episode(self, frames, agent_box, object_box):
        n = len(frames)
        w, h = frames[0].size
        fill = 0 if object_box is None else 200
        return {
            "agent": [Image.new("L", (w, h), 255) for _ in range(n)],
            "object": [Image.new("L", (w, h), fill) for _ in range(n)],
        }


def _row(sentinel, term=False, trunc=False):
    return {
        "observation_distracted": Image.new("RGB", (8, 8), (10, 20, 30)),
        "observation": Image.new("RGB", (8, 8), (1, 2, 3)),
        "state": [0.1, 0.2],
        "reward": 0.5,
        "terminated": term,
        "truncated": trunc,
        "sentinel": sentinel,
    }


def test_iter_episodes_splits_on_terminated():
    rows = [_row("a"), _row("b", term=True), _row("c"), _row("d", term=True)]
    episodes = list(iter_episodes(rows))
    assert [len(e) for e in episodes] == [2, 2]


def test_iter_episodes_splits_on_truncated():
    rows = [_row("a"), _row("b", trunc=True), _row("c", trunc=True)]
    episodes = list(iter_episodes(rows))
    assert [len(e) for e in episodes] == [2, 1]


def test_iter_episodes_emits_trailing_incomplete_episode():
    rows = [_row("a", term=True), _row("b"), _row("c")]  # last episode never ends
    episodes = list(iter_episodes(rows))
    assert [len(e) for e in episodes] == [1, 2]


_ORIG_COLS = ["observation_distracted", "observation", "state", "reward",
              "terminated", "truncated", "sentinel"]


def test_extract_episode_passes_through_original_columns_and_appends_preds():
    rows = [_row("a"), _row("b", term=True)]
    out, meta = extract_episode(
        rows, _FakeDetector(), _FakePredictor(), _CFG,
        obs_col="observation_distracted", orig_columns=_ORIG_COLS,
    )
    # Every original column is preserved verbatim, in order.
    assert out["sentinel"] == ["a", "b"]
    assert out["state"] == [[0.1, 0.2], [0.1, 0.2]]
    assert out["reward"] == [0.5, 0.5]
    assert out["terminated"] == [False, True]
    assert out["observation_distracted"] is not None and len(out["observation_distracted"]) == 2
    # Two new prediction columns, aligned per frame.
    assert len(out["pred_mask"]) == 2 and len(out["pred_object_mask"]) == 2
    assert all(np.asarray(m).max() == 255 for m in out["pred_mask"])
    assert all(np.asarray(m).max() == 200 for m in out["pred_object_mask"])
    assert meta["detected"] is True
    assert meta["n_frames"] == 2


def test_extract_episode_records_detection_failure_with_empty_object_masks():
    rows = [_row("a", term=True)]
    out, meta = extract_episode(
        rows, _FakeDetector(detected=False), _FakePredictor(), _CFG,
        obs_col="observation_distracted", orig_columns=_ORIG_COLS,
    )
    assert meta["detected"] is False
    assert meta["object_box"] is None
    assert all(np.asarray(m).sum() == 0 for m in out["pred_object_mask"])
