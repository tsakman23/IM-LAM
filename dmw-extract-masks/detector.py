"""Grounding DINO open-vocabulary object detection (HF transformers).

Pure helpers (`select_best_box`, `detect_with_fallback`) are unit-tested without
loading any model; `GroundingDinoDetector` wraps the actual checkpoint and is
covered by GPU integration tests.
"""
from typing import Callable, Optional

import numpy as np


def select_best_box(boxes, scores) -> Optional[list]:
    """Return the highest-scoring box as a list of floats, or None if empty."""
    scores = np.asarray(scores)
    if scores.shape[0] == 0:
        return None
    boxes = np.asarray(boxes)
    return boxes[int(np.argmax(scores))].tolist()


def select_object_box(boxes, scores, region=None, max_area=None):
    """Highest-scoring box subject to a workspace prior: keep only candidates
    whose center lies inside ``region`` (xyxy) and whose area is <= ``max_area``.
    Returns the surviving top-1 box as a list, or None if none survive.

    This encodes that the manipulated object is small and on the table, which
    rejects the two open-vocabulary failure modes seen on DMW: latching onto the
    large arm (too big) or a same-colored background distractor (off-table)."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape[0] == 0:
        return None
    boxes = np.asarray(boxes, dtype=float)
    keep = np.ones(scores.shape[0], dtype=bool)
    if region is not None:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        rx0, ry0, rx1, ry1 = region
        keep &= (cx >= rx0) & (cx <= rx1) & (cy >= ry0) & (cy <= ry1)
    if max_area is not None:
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep &= area <= max_area
    if not keep.any():
        return None
    idx = np.where(keep)[0]
    return boxes[idx[int(np.argmax(scores[idx]))]].tolist()


def detect_with_fallback(
    detect_fn: Callable,
    frame,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    fallback_box_threshold: float,
    fallback_text_threshold: float,
) -> tuple[Optional[list], bool]:
    """Detect at primary thresholds; if nothing is found, retry once at the
    fallback thresholds. Returns ``(box_or_None, detected)``.

    ``detect_fn(frame, prompt, box_threshold, text_threshold) -> box | None`` is
    injected so this policy is testable without a model.
    """
    box = detect_fn(frame, prompt, box_threshold, text_threshold)
    if box is not None:
        return box, True
    box = detect_fn(frame, prompt, fallback_box_threshold, fallback_text_threshold)
    return box, box is not None


class GroundingDinoDetector:
    """Thin wrapper around a HF zero-shot object detector. Loads the checkpoint
    lazily in ``__init__`` so importing this module stays cheap for unit tests."""

    def __init__(self, model_id: str, device: str, dtype=None,
                 object_region=None, object_max_box_area=None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        # Workspace prior for object detection (both optional): keep only detections
        # whose center is in `object_region` (xyxy px) and whose box area <=
        # `object_max_box_area` (px). None disables that constraint.
        self.object_region = object_region
        self.object_max_box_area = object_max_box_area
        self.processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        if dtype is not None:
            model = model.to(device, dtype=dtype)
        else:
            model = model.to(device)
        self.model = model.eval()

    def detect(self, frame, prompt: str, box_threshold: float, text_threshold: float) -> Optional[list]:
        """Return the highest-scoring box (xyxy, pixel coords) for ``prompt`` on
        ``frame`` (a PIL RGB image), or None if nothing passes the thresholds."""
        import torch

        # GDINO wants lowercase phrases split on periods, nested one level per image.
        phrases = [p.strip() for p in prompt.lower().split(".") if p.strip()]
        inputs = self.processor(images=frame, text=[phrases], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[frame.size[::-1]],  # (H, W)
        )[0]
        return select_object_box(
            results["boxes"].cpu().numpy(), results["scores"].cpu().numpy(),
            region=self.object_region, max_area=self.object_max_box_area,
        )

    def detect_object(
        self,
        frame,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        fallback_box_threshold: float,
        fallback_text_threshold: float,
    ) -> tuple[Optional[list], bool]:
        """Frame-0 object detection with the retry-at-lower-threshold policy."""
        return detect_with_fallback(
            self.detect, frame, prompt,
            box_threshold, text_threshold,
            fallback_box_threshold, fallback_text_threshold,
        )
