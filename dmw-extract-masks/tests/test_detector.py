import numpy as np

from detector import apply_box_offset, detect_with_fallback, select_best_box, select_object_box


def test_apply_box_offset_none_offset_is_noop():
    assert apply_box_offset([10, 10, 20, 20], None, 128, 128) == [10, 10, 20, 20]
    assert apply_box_offset([10, 10, 20, 20], [0, 0, 0, 0], 128, 128) == [10, 10, 20, 20]


def test_apply_box_offset_shifts_each_edge():
    # dx0, dy0, dx1, dy1 added to the corresponding edges (used to turn a detected
    # cabinet box into the door+handle box on door-open).
    assert apply_box_offset([29, 45, 72, 81], [14, 5, 16, 0], 128, 128) == [43, 50, 88, 81]


def test_apply_box_offset_clamps_to_image_bounds():
    assert apply_box_offset([120, 120, 125, 125], [0, 0, 20, 20], 128, 128) == [120, 120, 128, 128]
    assert apply_box_offset([5, 5, 20, 20], [-10, -10, 0, 0], 128, 128) == [0, 0, 20, 20]


def test_apply_box_offset_none_box_returns_none():
    assert apply_box_offset(None, [14, 5, 16, 0], 128, 128) is None


def test_select_object_box_no_filters_is_top1():
    boxes = np.array([[0, 0, 4, 4], [10, 10, 14, 14]], dtype=np.float32)
    scores = np.array([0.3, 0.7], dtype=np.float32)
    assert select_object_box(boxes, scores) == [10.0, 10.0, 14.0, 14.0]


def test_select_object_box_region_rejects_out_of_region_top_score():
    # Top-scoring box's center is outside the workspace; the in-region one wins.
    boxes = np.array([[0, 0, 4, 4], [50, 60, 60, 70]], dtype=np.float32)  # centers (2,2), (55,65)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    assert select_object_box(boxes, scores, region=[40, 50, 80, 90]) == [50.0, 60.0, 60.0, 70.0]


def test_select_object_box_max_area_rejects_large_box():
    # Top-scoring box is huge (e.g. the arm); the small one wins.
    boxes = np.array([[0, 0, 100, 100], [40, 40, 50, 50]], dtype=np.float32)  # areas 10000, 100
    scores = np.array([0.9, 0.5], dtype=np.float32)
    assert select_object_box(boxes, scores, max_area=2500) == [40.0, 40.0, 50.0, 50.0]


def test_select_object_box_returns_none_when_no_candidate_survives():
    boxes = np.array([[0, 0, 4, 4]], dtype=np.float32)  # center (2,2) outside region
    scores = np.array([0.9], dtype=np.float32)
    assert select_object_box(boxes, scores, region=[40, 50, 80, 90]) is None


def test_select_best_box_returns_none_when_no_detections():
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,), dtype=np.float32)
    assert select_best_box(boxes, scores) is None


def test_select_best_box_picks_highest_scoring():
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], dtype=np.float32)
    scores = np.array([0.2, 0.5, 0.4], dtype=np.float32)
    assert select_best_box(boxes, scores) == [2.0, 2.0, 3.0, 3.0]


class _StubDetect:
    """Records calls and returns a box keyed on the box_threshold used."""

    def __init__(self, by_box_threshold):
        self.by_box_threshold = by_box_threshold
        self.calls = []

    def __call__(self, frame, prompt, box_threshold, text_threshold):
        self.calls.append((box_threshold, text_threshold))
        return self.by_box_threshold.get(box_threshold)


def test_detect_with_fallback_uses_primary_when_found():
    detect = _StubDetect({0.3: [1, 2, 3, 4]})
    box, detected = detect_with_fallback(
        detect, frame=None, prompt="p",
        box_threshold=0.3, text_threshold=0.25,
        fallback_box_threshold=0.15, fallback_text_threshold=0.15,
    )
    assert box == [1, 2, 3, 4]
    assert detected is True
    assert detect.calls == [(0.3, 0.25)]  # fallback not attempted


def test_detect_with_fallback_retries_at_lower_threshold():
    detect = _StubDetect({0.15: [5, 6, 7, 8]})  # primary (0.3) misses, fallback hits
    box, detected = detect_with_fallback(
        detect, frame=None, prompt="p",
        box_threshold=0.3, text_threshold=0.25,
        fallback_box_threshold=0.15, fallback_text_threshold=0.15,
    )
    assert box == [5, 6, 7, 8]
    assert detected is True
    assert detect.calls == [(0.3, 0.25), (0.15, 0.15)]


def test_detect_with_fallback_returns_none_when_both_fail():
    detect = _StubDetect({})  # nothing detected at any threshold
    box, detected = detect_with_fallback(
        detect, frame=None, prompt="p",
        box_threshold=0.3, text_threshold=0.25,
        fallback_box_threshold=0.15, fallback_text_threshold=0.15,
    )
    assert box is None
    assert detected is False
    assert len(detect.calls) == 2
