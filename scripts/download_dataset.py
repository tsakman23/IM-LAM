#!/usr/bin/env python
"""Pre-download + build a DMW dataset into the local cache, so `run_slapo.py`
just cache-hits instead of downloading/building at train time.

Why a separate script:
  - Downloads run standalone (no CUDA / DataLoader workers / EGL), which also
    sidesteps the fork-thread deadlock that can wedge the FIRST-TIME dataset
    build when it happens inside the training process.
  - Safe to Ctrl+C and rerun: HF writes each shard to a `*.incomplete` temp and
    only atomically promotes it to the cache once verified, so completed shards
    are reused and only the interrupted shard is re-fetched. No corruption.

It reuses the training loader (`ifo.common.utils.data.get_dataset`) with the SAME
arguments the run uses, so the cache it fills is exactly the one training reads.
IMPORTANT: pass the SAME --cache-dir you give the run as `dataset.cache_dir`.

Examples
--------
# MaskLAM data (authors' release, agent mask only):
python scripts/download_dataset.py --env Meta-World/masked-MT1-push-wall-v3

# Foreground-MaskLAM data (my repo, with object masks), both splits:
python scripts/download_dataset.py \
    --env Meta-World/masked-MT1-push-v3 \
    --repo tsakman23/visual_masked_distracting_metaworld --with-object-mask
"""
import argparse
import os
import time

os.environ.setdefault("HF_HOME", "/data2/masklam/datasets/hf_home")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, help="Env/dataset name, e.g. Meta-World/masked-MT1-push-v3")
    p.add_argument("--split", default="both", choices=["train", "test", "both"])
    p.add_argument("--cache-dir", default=None,
                   help="Dataset cache dir; MUST match the run's dataset.cache_dir. "
                        "Default: ./datasets/slapo/<env>")
    p.add_argument("--repo", default=None,
                   help="HF repo override (Meta-World). Default = authors' EpicPinkPenguin release. "
                        "Use tsakman23/visual_masked_distracting_metaworld for object masks.")
    p.add_argument("--with-object-mask", action="store_true",
                   help="Also fetch the object_mask column (Foreground-MaskLAM; needs an object-mask repo).")
    p.add_argument("--block-size", type=int, default=13,
                   help="Matches the stage config (DMW stage-1 = 13). Does not affect what is cached.")
    p.add_argument("--retries", type=int, default=5, help="Retries per split on transient errors.")
    args = p.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    cache_dir = args.cache_dir or f"./datasets/slapo/{args.env}"

    # Imported here so `--help` is instant and env vars are set first.
    from ifo.common.utils.data import get_dataset

    print(f"Priming cache for {args.env}")
    print(f"  repo       : {args.repo or 'EpicPinkPenguin/visual_distracting_metaworld (default)'}")
    print(f"  object_mask: {args.with_object_mask}")
    print(f"  cache_dir  : {cache_dir}")
    print(f"  HF_HOME    : {os.environ['HF_HOME']}\n")

    for split in splits:
        for attempt in range(1, args.retries + 1):
            try:
                t0 = time.time()
                ds = get_dataset(
                    name=args.env, split=split, cache_dir=cache_dir,
                    block_size=args.block_size, dataset_path=args.repo,
                    with_object_mask=args.with_object_mask,
                )
                print(f"[{split}] ready: {len(ds):,} rows in {time.time() - t0:.0f}s")
                break
            except KeyboardInterrupt:
                print(f"\n[{split}] interrupted - completed shards are kept; rerun to resume.")
                raise
            except Exception as e:  # transient network / build errors: retry with backoff
                print(f"[{split}] attempt {attempt}/{args.retries} failed: {type(e).__name__}: {str(e)[:200]}")
                if attempt == args.retries:
                    raise
                time.sleep(5 * attempt)

    print("\nDone - run_slapo.py will now cache-hit for these splits.")


if __name__ == "__main__":
    main()
