"""Compare two column .npz fixtures: bit-equality per column, else stats."""
import argparse

import numpy as np

IMG_COLS = ("observation", "observation_distracted", "mask")
VEC_COLS = ("state", "action")
SCALAR_COLS = ("reward", "terminated", "truncated")


def compare(path_a: str, path_b: str, n: int) -> dict:
    a, b = np.load(path_a), np.load(path_b)
    m = min(n, len(a["observation"]), len(b["observation"]))
    report = {}
    for col in IMG_COLS + VEC_COLS + SCALAR_COLS:
        xa, xb = a[col][:m], b[col][:m]
        if xa.shape != xb.shape:
            report[col] = f"SHAPE MISMATCH {xa.shape} vs {xb.shape}"
            continue
        identical = bool(np.array_equal(xa, xb))
        if identical:
            report[col] = "IDENTICAL"
        else:
            if col in SCALAR_COLS:
                report[col] = f"DIFF (mean_a={np.mean(xa):.4f} mean_b={np.mean(xb):.4f})"
            else:
                diff = np.abs(xa.astype(np.float64) - xb.astype(np.float64))
                report[col] = f"DIFF (max_abs={diff.max():.4f} mean_abs={diff.mean():.6f})"
    for k, v in report.items():
        print(f"{k:26s} {v}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--n", type=int, default=200)
    compare(p.parse_args().a, p.parse_args().b, p.parse_args().n)
