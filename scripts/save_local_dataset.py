#!/usr/bin/env python
"""Make a local, training-loadable (save_to_disk) copy of a HuggingFace dataset.

Path B for existing data: no regeneration. Loads a dataset from HF (or its already-built
local cache) once, then writes it in ``save_to_disk`` Arrow format to
``<out-root>/<task>/<split>`` - the exact layout ``ifo.common.data.huggingface`` reads via
``load_from_disk`` when a run's ``dataset.dataset_path`` points at ``<out-root>``. After this,
training memory-maps the Arrow directly: no parquet->Arrow build, no HF fingerprint churn.

Works for BOTH methods:
  - MaskLAM baseline: --repo EpicPinkPenguin/visual_distracting_metaworld (authors' release,
    agent mask only). Not regenerated - just mirrored locally.
  - Foreground-MaskLAM: --repo tsakman23/visual_masked_distracting_metaworld (object masks).

The raw dataset (all columns, incl. observation_distracted / object_mask) is saved as-is;
MetaWorldDataset re-applies its rename / column selection at load time.

Example
-------
# MaskLAM data for sweep-into (both splits):
python scripts/save_local_dataset.py \
  --repo EpicPinkPenguin/visual_distracting_metaworld --task sweep-into-v3 \
  --splits train test --out-root ./datasets/slapo_local

# Foreground-MaskLAM object-mask data for push:
python scripts/save_local_dataset.py \
  --repo tsakman23/visual_masked_distracting_metaworld --task push-v3 \
  --splits train test --out-root ./datasets/slapo_local
"""
import argparse
import os
import shutil
import time


def local_dest(out_root: str, task: str, split: str) -> str:
    """Local layout for a (task, split) copy: ``<out_root>/<task>/<split>``."""
    return os.path.join(out_root, task, split)


def _is_complete(dest: str) -> bool:
    """True if ``dest`` holds a finished ``save_to_disk`` dataset (has dataset_info.json)."""
    return os.path.isdir(dest) and os.path.exists(os.path.join(dest, "dataset_info.json"))


def write_local_copy(ds, dest: str, skip_if_exists: bool = True) -> bool:
    """Write ``ds`` to ``dest`` in ``save_to_disk`` (load_from_disk-able) format.

    Idempotent: if a completed copy already exists and ``skip_if_exists`` is True, do nothing
    and return False (avoids duplicating / stubbing on re-runs). Otherwise (fresh data, or an
    interrupted partial copy) replace it and return True.

    Args:
        ds: A ``datasets.Dataset`` to persist.
        dest (str): Target directory (typically ``local_dest(...)``).
        skip_if_exists (bool): Skip when a completed copy is already present.

    Returns:
        bool: True if written, False if skipped.
    """
    if skip_if_exists and _is_complete(dest):
        return False
    if os.path.isdir(dest):
        shutil.rmtree(dest)  # clear a partial/older copy so save_to_disk starts clean
    ds.save_to_disk(dest)
    return True


def save_split_local(
    repo: str, task: str, split: str, out_root: str,
    cache_dir: str = None, skip_if_exists: bool = True,
) -> str:
    """Load one (repo, task, split) from HF (or its built cache) and write a local copy.

    Returns the destination path. Skips the load entirely when a completed copy already exists.
    """
    dest = local_dest(out_root, task, split)
    if skip_if_exists and _is_complete(dest):
        print(f"[{split}] already present at {dest} - skipping")
        return dest
    from datasets import load_dataset  # imported here so --help / import stays light

    cache_dir = cache_dir or f"./datasets/slapo/Meta-World/masked-MT1-{task}"
    t0 = time.time()
    print(f"[{split}] loading {repo}:{task} (cache_dir={cache_dir}) ...")
    ds = load_dataset(path=repo, name=task, split=split, cache_dir=cache_dir)
    print(f"[{split}] loaded {len(ds):,} rows, columns={ds.column_names} in {time.time() - t0:.0f}s")
    write_local_copy(ds, dest, skip_if_exists=False)
    print(f"[{split}] save_to_disk -> {dest}  (total {time.time() - t0:.0f}s)")
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="HF dataset repo id, e.g. tsakman23/visual_masked_distracting_metaworld")
    p.add_argument("--task", required=True, help="Bare HF config / task slug, e.g. push-v3 (no MT1-/masked- prefix)")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--out-root", default="./datasets/slapo_local",
                   help="Root for local copies; layout <out-root>/<task>/<split>. Point the run's "
                        "dataset.dataset_path at this. Use /tmp/slapo_local for RAM-fast (ephemeral) reads.")
    p.add_argument("--cache-dir", default=None,
                   help="HF load cache dir. Default reuses the training path ./datasets/slapo/Meta-World/"
                        "masked-MT1-<task> so an already-built split cache-hits instead of rebuilding.")
    args = p.parse_args()

    print(f"repo={args.repo}  task={args.task}  splits={args.splits}  out_root={args.out_root}\n")
    for split in args.splits:
        save_split_local(args.repo, args.task, split, args.out_root, cache_dir=args.cache_dir)
    print(f"\nLocal copies ready under {args.out_root}. Train with dataset.dataset_path={args.out_root}")


if __name__ == "__main__":
    main()
