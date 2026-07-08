"""Re-push an already-generated split to the Hub from its local from_generator cache,
WITHOUT regenerating. Use when generation succeeded but push_to_hub failed (e.g. auth).

Requires HF auth: `hf auth login` (write token) with HF_HOME set, or `export HF_TOKEN=...`.

Usage:
  HF_HOME=/data2/masklam/datasets/hf_home PYTHONPATH=. \
  python verify/repush_from_cache.py --config push-v3 --split train
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("HF_HOME", "/data2/masklam/datasets/hf_home")
# Make the repo root importable so `python verify/repush_from_cache.py` works
# from anywhere without needing PYTHONPATH=.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset, concatenate_datasets  # noqa: E402
from generate_dataset_huggingface import REPO_ID  # noqa: E402


def main(config: str, split: str, private: bool) -> None:
    pattern = f"datasets/hf_cache/{config}_{split}/generator/default-*/0.0.0/*.arrow"
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise SystemExit(f"No cached shards found at {pattern} - has this split been generated?")
    print(f"Loading {len(shards)} cached shards for {config}/{split} (no regeneration)...")
    ds = concatenate_datasets([Dataset.from_file(s) for s in shards])
    print(f"  {len(ds):,} rows; columns={ds.column_names}")
    print(f"Pushing to {REPO_ID}  config={config}  split={split}  private={private} ...")
    ds.push_to_hub(REPO_ID, config_name=config, split=split, private=private)
    print(f"Done: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="push-v3")
    p.add_argument("--split", required=True, choices=["train", "test"])
    p.add_argument("--private", action="store_true")
    a = p.parse_args()
    main(a.config, a.split, a.private)
