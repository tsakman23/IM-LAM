import numpy as np
from PIL import Image

from eval_masks import binary_iou, iou_over_frames, summarize_iou, to_binary


def test_to_binary_from_L_mask_thresholds_above_zero():
    m = Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8), "L")
    np.testing.assert_array_equal(to_binary(m), np.array([[False, True], [True, False]]))


def test_binary_iou_identical_is_one():
    a = np.array([[1, 1], [0, 0]], dtype=bool)
    assert binary_iou(a, a) == 1.0


def test_binary_iou_disjoint_is_zero():
    a = np.array([[1, 0], [0, 0]], dtype=bool)
    b = np.array([[0, 1], [0, 0]], dtype=bool)
    assert binary_iou(a, b) == 0.0


def test_binary_iou_half_overlap():
    # pred covers 2 px, gt covers 2 px, they share 1 -> inter 1 / union 3.
    pred = np.array([[1, 1], [0, 0]], dtype=bool)
    gt = np.array([[1, 0], [1, 0]], dtype=bool)
    assert binary_iou(pred, gt) == 1 / 3


def test_binary_iou_both_empty_is_one_by_convention():
    z = np.zeros((2, 2), dtype=bool)
    assert binary_iou(z, z) == 1.0


def test_binary_iou_pred_empty_gt_nonempty_is_zero():
    pred = np.zeros((2, 2), dtype=bool)
    gt = np.array([[1, 0], [0, 0]], dtype=bool)
    assert binary_iou(pred, gt) == 0.0


def test_iou_over_frames_aligns_pred_and_gt():
    preds = [np.array([[1, 0]], dtype=bool), np.array([[1, 1]], dtype=bool)]
    gts = [np.array([[1, 0]], dtype=bool), np.array([[1, 0]], dtype=bool)]
    assert iou_over_frames(preds, gts) == [1.0, 0.5]


def test_summarize_iou_reports_mean_median_and_area_weighted():
    ious = [0.0, 1.0, 0.5]
    areas = [0, 10, 10]  # first frame has no object -> zero weight
    s = summarize_iou(ious, areas)
    assert s["n"] == 3
    assert s["mean"] == np.mean(ious)
    assert s["median"] == 0.5
    # area-weighted ignores the empty-object frame: (1.0*10 + 0.5*10)/20 = 0.75
    assert s["area_weighted"] == 0.75
