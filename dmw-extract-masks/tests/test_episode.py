import numpy as np
from PIL import Image

from episode import empty_masks, mask_tensor_to_pil


def test_empty_masks_returns_n_black_L_images():
    masks = empty_masks(3, 8, 6)
    assert len(masks) == 3
    for m in masks:
        assert isinstance(m, Image.Image)
        assert m.mode == "L"
        assert m.size == (6, 8)  # PIL size is (W, H)
        assert np.asarray(m).sum() == 0


def test_mask_tensor_to_pil_binarizes_to_0_255():
    arr = np.array([[1, 0], [0, 1]], dtype=np.float32)
    img = mask_tensor_to_pil(arr)
    assert img.mode == "L"
    assert img.size == (2, 2)
    np.testing.assert_array_equal(np.asarray(img), np.array([[255, 0], [0, 255]], dtype=np.uint8))


def test_mask_tensor_to_pil_squeezes_leading_singleton_dims():
    # SAM2 post-processed masks arrive as (1, 1, H, W) per object.
    arr = np.zeros((1, 1, 4, 5), dtype=np.float32)
    arr[0, 0, 1, 2] = 1.0
    img = mask_tensor_to_pil(arr)
    assert img.size == (5, 4)  # (W, H)
    out = np.asarray(img)
    assert out[1, 2] == 255
    assert out.sum() == 255
