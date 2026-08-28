"""Contact-sheet rendering for qualitative mask inspection.

Pure helpers (`select_timesteps`, `overlay_mask`, `make_grid`, `draw_box`) are
unit-tested; `render_episode_sheet` composes them into a per-episode panel.
"""
import numpy as np
from PIL import Image, ImageDraw


def select_timesteps(n: int, k: int = 5) -> list:
    """Pick up to ``k`` evenly spaced frame indices including 0 and n-1."""
    if n <= 0:
        return []
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return sorted(set(int(i) for i in idx))


def overlay_mask(frame, mask, color=(255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend ``color`` onto ``frame`` (H,W,3 uint8) where ``mask`` is True."""
    frame = np.asarray(frame).astype(np.float32)
    mask = np.asarray(mask, dtype=bool)
    out = frame.copy()
    out[mask] = (1.0 - alpha) * frame[mask] + alpha * np.asarray(color, dtype=np.float32)
    return out.astype(np.uint8)


def draw_box(frame, box, color=(0, 255, 0), width: int = 1) -> np.ndarray:
    """Draw a rectangle outline (xyxy) on a copy of ``frame``. ``box=None`` is a
    no-op (used for detection failures)."""
    frame = np.asarray(frame).astype(np.uint8)
    if box is None:
        return frame.copy()
    img = Image.fromarray(frame).convert("RGB")
    drawer = ImageDraw.Draw(img)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    drawer.rectangle([x0, y0, x1, y1], outline=tuple(color), width=width)
    return np.asarray(img)


def render_episode_sheet(frames, agent_pred, object_pred, agent_gt, object_gt,
                         object_box, timesteps=None) -> Image.Image:
    """One panel per episode: rows are sampled timesteps, columns are
    [distracted | frame-0 box | agent pred | object pred | agent GT | object GT].
    Composed entirely from the unit-tested helpers above."""
    n = len(frames)
    if timesteps is None:
        timesteps = select_timesteps(n, 5)

    def _rgb(x):
        return np.asarray(Image.fromarray(np.asarray(x)).convert("RGB")).astype(np.uint8)

    rows = []
    for t in timesteps:
        base = _rgb(frames[t])
        boxed = draw_box(base, object_box if t == timesteps[0] else None, (0, 255, 0), 1)
        row = [
            Image.fromarray(base),
            Image.fromarray(boxed),
            Image.fromarray(overlay_mask(base, np.asarray(agent_pred[t]) > 0, (255, 0, 0))),
            Image.fromarray(overlay_mask(base, np.asarray(object_pred[t]) > 0, (0, 128, 255))),
            Image.fromarray(overlay_mask(base, np.asarray(agent_gt[t]) > 0, (255, 0, 0))),
            Image.fromarray(overlay_mask(base, np.asarray(object_gt[t]) > 0, (0, 128, 255))),
        ]
        rows.append(row)
    return make_grid(rows, pad=2, bg=(30, 30, 30))


def make_grid(rows, pad: int = 0, bg=(255, 255, 255)) -> Image.Image:
    """Tile a 2-D list of equal-size PIL images into one grid image."""
    n_rows = len(rows)
    n_cols = len(rows[0])
    cw, ch = rows[0][0].size
    grid_w = n_cols * cw + (n_cols + 1) * pad
    grid_h = n_rows * ch + (n_rows + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), bg)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x = pad + c * (cw + pad)
            y = pad + r * (ch + pad)
            grid.paste(cell, (x, y))
    return grid
