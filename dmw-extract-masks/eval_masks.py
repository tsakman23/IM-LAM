"""IoU evaluation of predicted masks against the ground-truth simulator masks
that live in the same superset rows. Pure metric helpers are unit-tested; the
shard evaluator and CSV writer operate on a produced superset dataset.
"""
import numpy as np


def to_binary(mask) -> np.ndarray:
    """Coerce a PIL image / ndarray mask to a boolean array (foreground = > 0)."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.any(axis=-1)
    return arr > 0


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks. Both empty (union 0) is
    treated as perfect agreement (1.0)."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float(inter) / float(union)


def iou_over_frames(preds, gts) -> list:
    """Per-frame IoU for aligned lists of predicted and GT masks."""
    return [binary_iou(to_binary(p), to_binary(g)) for p, g in zip(preds, gts)]


def summarize_iou(ious, areas=None) -> dict:
    """Summary stats over a list of per-frame IoUs. If ``areas`` (object pixel
    counts) are given, also report an area-weighted mean that ignores frames
    with no object."""
    ious = np.asarray(ious, dtype=float)
    out = {
        "n": int(ious.size),
        "mean": float(np.mean(ious)) if ious.size else float("nan"),
        "median": float(np.median(ious)) if ious.size else float("nan"),
        "p10": float(np.percentile(ious, 10)) if ious.size else float("nan"),
        "p90": float(np.percentile(ious, 90)) if ious.size else float("nan"),
    }
    if areas is not None:
        areas = np.asarray(areas, dtype=float)
        total = areas.sum()
        out["area_weighted"] = float((ious * areas).sum() / total) if total > 0 else float("nan")
    return out
