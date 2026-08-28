import numpy as np
from PIL import Image

from contact_sheet import draw_box, make_grid, overlay_mask, select_timesteps


def test_select_timesteps_includes_endpoints_and_quartiles():
    assert select_timesteps(10, 5) == [0, 2, 4, 7, 9]


def test_select_timesteps_dedups_for_short_episodes():
    assert select_timesteps(2, 5) == [0, 1]
    assert select_timesteps(1, 5) == [0]


def test_overlay_mask_blends_masked_pixels_and_leaves_others():
    frame = np.full((2, 2, 3), 100, dtype=np.uint8)
    mask = np.array([[True, False], [False, False]])
    out = overlay_mask(frame, mask, color=(255, 0, 0), alpha=0.5)
    assert out.shape == (2, 2, 3)
    np.testing.assert_array_equal(out[0, 0], [177, 50, 50])  # blended toward red
    np.testing.assert_array_equal(out[0, 1], [100, 100, 100])  # untouched


def test_draw_box_colors_border_and_leaves_outside():
    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    out = draw_box(frame, [1, 1, 4, 4], color=(0, 255, 0), width=1)
    assert out.shape == (6, 6, 3)
    np.testing.assert_array_equal(out[1, 1], [0, 255, 0])  # box corner drawn
    np.testing.assert_array_equal(out[0, 0], [0, 0, 0])    # outside untouched


def test_draw_box_with_none_box_is_noop():
    frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    out = draw_box(frame, None, color=(0, 255, 0))
    np.testing.assert_array_equal(out, frame)


def test_make_grid_tiles_images_into_rows_and_columns():
    cell = Image.new("RGB", (5, 4), (0, 0, 0))  # (W=5, H=4)
    grid = make_grid([[cell, cell, cell], [cell, cell, cell]])  # 2 rows x 3 cols
    assert isinstance(grid, Image.Image)
    assert grid.size == (15, 8)  # (3*5, 2*4)
