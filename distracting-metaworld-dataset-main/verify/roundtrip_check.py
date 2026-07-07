"""Build a small dataset via Dataset.from_generator and assert schema + parity."""
import argparse
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/data2/masklam/datasets/hf_home")

from datasets import Dataset  # noqa: E402
from generate_dataset_huggingface import FEATURES, episode_generator  # noqa: E402

EXPECTED = {
    "observation", "observation_distracted", "mask", "object_mask",
    "state", "object_state", "action", "reward", "terminated", "truncated",
}


def check(env_name: str, golden: str, n: int) -> None:
    ds = Dataset.from_generator(
        episode_generator, features=FEATURES,
        gen_kwargs=dict(
            env_name=env_name, split="train",
            max_num_episodes=None, max_num_steps=n,
            base_seed=0, camera_name="corner",
            background_dataset_path="datasets/DAVIS",
        ),
        cache_dir="datasets/hf_cache/roundtrip_tmp",
    )
    cols = set(ds.column_names)
    assert EXPECTED.issubset(cols), f"missing columns: {EXPECTED - cols}"
    g = np.load(golden)
    m = min(n, len(g["observation"]), len(ds))
    for i in range(m):
        row = ds[i]
        assert np.array_equal(np.asarray(row["observation"].convert("RGB"), np.uint8), g["observation"][i]), f"obs mismatch @ {i}"
        assert np.array_equal(np.asarray(row["mask"].convert("L"), np.uint8), g["mask"][i]), f"mask mismatch @ {i}"
        assert np.allclose(np.asarray(row["state"], np.float32), g["state"][i]), f"state mismatch @ {i}"
        assert np.allclose(np.asarray(row["action"], np.float32), g["action"][i]), f"action mismatch @ {i}"
    # object columns are present and well-formed
    assert np.asarray(ds[0]["object_mask"].convert("L")).shape == (128, 128)
    assert len(ds[0]["object_state"]) == 7
    print(f"round-trip OK: {len(cols)} columns, {m} rows matched golden")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Meta-World/MT1-push-v3")
    p.add_argument("--golden", default="datasets/verify/golden_push-v3_seed0.npz")
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    check(a.env, a.golden, a.n)
