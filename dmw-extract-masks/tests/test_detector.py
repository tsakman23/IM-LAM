import numpy as np

from detector import detect_with_fallback, select_best_box


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
