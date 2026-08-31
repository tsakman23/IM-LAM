#!/usr/bin/env python
"""Full-scale mask-extraction runner for one task.

Streams the source DMW dataset, extracts agent+object masks per episode, and
writes superset shards (all original columns verbatim + pred_mask/pred_object_mask).
Optionally consolidates to a loadable save_to_disk copy and/or pushes to the Hub.

Examples
--------
# Proof: 40 episodes from local cache -> loadable local superset
python main.py push-v3 --splits train --source-local --max-episodes 40 \
    --save-local /data2/masklam/datasets/dmw_sam_local

# Full run, stream from hub, push a superset repo
python main.py push-v3 --splits train test \
    --push-to-hub tsakman23/visual_masked_distracting_metaworld_sam
"""
import argparse
import glob
import json
import logging
import os

from datasets import Dataset

from utils import resolve_task_config
from worker import run_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

_SNAP = ("/data2/masklam/datasets/hf_home/hub/"
         "datasets--tsakman23--visual_masked_distracting_metaworld/snapshots")


def _local_source(task, split):
    files = sorted(glob.glob(f"{_SNAP}/*/{task}/{split}-*.parquet"))
    # The HF cache may hold several snapshot dirs, each with the SAME shard files;
    # dedupe by shard basename so we don't process the data multiple times.
    unique = {}
    for f in files:
        unique.setdefault(os.path.basename(f), f)
    return sorted(unique.values())  # may be empty if the split isn't cached


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"))
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-episodes", type=int, default=None, help="cap episodes per split (debug)")
    ap.add_argument("--source-root", default=None, metavar="ROOT",
                    help="RECOMMENDED: read a staged save_to_disk copy at ROOT/<task>/<split> "
                         "(fast local read, no hub streaming; stage with scripts/save_local_dataset.py)")
    ap.add_argument("--source-local", action="store_true",
                    help="read the HF cache's parquet shards (deduped across snapshots) instead of streaming")
    ap.add_argument("--save-local", default=None, metavar="ROOT",
                    help="consolidate shards to ROOT/<task>/<split> via save_to_disk "
                         "(loadable by ifo get_metaworld_dataset with a local path)")
    ap.add_argument("--push-to-hub", default=None, metavar="REPO")
    ap.add_argument("--push-only", action="store_true",
                    help="skip extraction; push the already-generated save_to_disk copy "
                         "(from --save-local ROOT/<task>/<split>) to --push-to-hub. No GPU.")
    args = ap.parse_args()

    config = json.load(open(args.config))
    resolve_task_config(config, args.task)  # fail fast if the task is unconfigured

    if args.push_only:
        if not (args.save_local and args.push_to_hub):
            raise SystemExit("--push-only requires --save-local (source) and --push-to-hub (target)")
        from datasets import load_from_disk
        for split in args.splits:
            src = os.path.join(args.save_local, args.task, split)
            if not os.path.isdir(src):
                raise SystemExit(f"--push-only: {src} not found (generate it first with --save-local)")
            print(f"[{args.task}/{split}] pushing {src} -> {args.push_to_hub} ...")
            load_from_disk(src).push_to_hub(args.push_to_hub, config_name=args.task, split=split)
            print(f"[{args.task}/{split}] pushed.")
        return

    processed = 0
    for split in args.splits:
        if args.source_root:
            source = os.path.join(args.source_root, args.task, split)
            if not os.path.isdir(source):
                logging.warning("skipping %s/%s: --source-root %s not found "
                                "(stage it with scripts/save_local_dataset.py)", args.task, split, source)
                continue
        elif args.source_local:
            source = _local_source(args.task, split)
            if not source:
                logging.warning("skipping %s/%s: no cached parquet found", args.task, split)
                continue
        else:
            source = None
        stats = run_worker(args.task, split, config, device=args.device,
                           source=source, max_episodes=args.max_episodes)
        processed += 1
        print(f"[{args.task}/{split}] episodes={stats['episodes']} frames={stats['frames']} "
              f"object_detection_rate={stats['detection_rate']:.3f} shards={len(stats['shards'])}")

        if not stats["shards"]:
            continue
        if args.save_local:
            out = os.path.join(args.save_local, args.task, split)
            Dataset.from_parquet(stats["shards"]).save_to_disk(out)
            print(f"[{args.task}/{split}] saved loadable copy -> {out}")
        if args.push_to_hub:
            Dataset.from_parquet(stats["shards"]).push_to_hub(
                args.push_to_hub, config_name=args.task, split=split)
            print(f"[{args.task}/{split}] pushed -> {args.push_to_hub} ({args.task}/{split})")

    if processed == 0:
        raise SystemExit(f"{args.task}: no splits were processed (all requested sources missing)")


if __name__ == "__main__":
    main()
