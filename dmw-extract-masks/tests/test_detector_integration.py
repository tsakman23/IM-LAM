"""GPU integration test for Grounding DINO detection on a real cached frame.
    pytest tests/test_detector_integration.py -m integration
"""
import glob
import io

import numpy as np
import pytest
import torch
from PIL import Image

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SNAP = ("/data2/masklam/datasets/hf_home/hub/"
        "datasets--tsakman23--visual_masked_distracting_metaworld/snapshots")
PROMPT = "a puck. a small cube. a red block."

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _first_push_frame():
    shards = glob.glob(f"{SNAP}/*/push-v3/train-00000-of-*.parquet")
    if not shards:
        pytest.skip("cached push-v3 parquet not found")
    import pyarrow.parquet as pq
    row = pq.read_table(shards[0], columns=["observation_distracted"]).slice(0, 1)
    item = row.column("observation_distracted").to_pylist()[0]
    return Image.open(io.BytesIO(item["bytes"])).convert("RGB")


@pytest.mark.integration
def test_detect_returns_in_bounds_box_for_puck():
    from detector import GroundingDinoDetector

    frame = _first_push_frame()
    det = GroundingDinoDetector(GDINO_ID, "cuda:0")
    box = det.detect(frame, PROMPT, box_threshold=0.15, text_threshold=0.1)
    assert box is not None and len(box) == 4
    x0, y0, x1, y1 = box
    W, H = frame.size
    assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H


@pytest.mark.integration
def test_detect_object_reports_detection_with_fallback():
    from detector import GroundingDinoDetector

    frame = _first_push_frame()
    det = GroundingDinoDetector(GDINO_ID, "cuda:0")
    box, detected = det.detect_object(
        frame, PROMPT,
        box_threshold=0.35, text_threshold=0.3,       # strict primary
        fallback_box_threshold=0.1, fallback_text_threshold=0.1,  # lenient fallback
    )
    assert detected is True
    assert box is not None and len(box) == 4
