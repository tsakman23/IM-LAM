#!/usr/bin/env python
"""Phase 3 driver: run the extractor on a few push-v3 episodes (real cached
data), render contact sheets, and compute agent/object IoU vs the GT masks in
the same rows plus the frame-0 object detection rate. Produces artifacts for
visual + quantitative inspection. Composes the unit-tested pipeline functions.

    python run_prototype.py [task] [n_episodes]
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from datasets import load_dataset

from contact_sheet import render_episode_sheet
from detector import GroundingDinoDetector
from episode import Sam2DualPredictor
from eval_masks import binary_iou, summarize_iou, to_binary
from utils import get_torch_dtype, resolve_task_config
from worker import extract_episode, iter_episodes

_SNAP = ("/data2/masklam/datasets/hf_home/hub/"
         "datasets--tsakman23--visual_masked_distracting_metaworld/snapshots")


def _source(task, split):
    files = sorted(glob.glob(f"{_SNAP}/*/{task}/{split}-00000-of-*.parquet"))
    if not files:
        raise SystemExit(f"no cached parquet for {task}/{split} under {_SNAP}")
    return files


def main(task="push-v3", n_episodes=6, split="train", device="cuda:0"):
    here = os.path.dirname(os.path.abspath(__file__))
    config = json.load(open(os.path.join(here, "config.json")))
    cfg = resolve_task_config(config, task)
    obs_col = cfg["observation_column"]

    ds = load_dataset("parquet", data_files=_source(task, split), split="train", streaming=True)
    orig_columns = list(ds.features.keys())

    detector = GroundingDinoDetector(
        cfg["grounding_dino_model_id"], device,
        object_region=cfg.get("object_region"),
        object_max_box_area=cfg.get("object_max_box_area"),
    )
    predictor = Sam2DualPredictor(cfg["sam2_model_id"], device, get_torch_dtype(cfg["dtype"]))

    out_dir = f"/data2/masklam/results/dmw_sam/{task}"
    os.makedirs(out_dir, exist_ok=True)

    per_ep, agent_all, object_all, area_all = [], [], [], []
    detected = 0

    for ep_i, ep_rows in enumerate(iter_episodes(ds)):
        if ep_i >= n_episodes:
            break
        out, meta = extract_episode(ep_rows, detector, predictor, cfg, obs_col, orig_columns)
        detected += int(meta["detected"])

        a_iou = [binary_iou(to_binary(p), to_binary(g)) for p, g in zip(out["pred_mask"], out["mask"])]
        o_iou = [binary_iou(to_binary(p), to_binary(g)) for p, g in zip(out["pred_object_mask"], out["object_mask"])]
        o_area = [int(to_binary(g).sum()) for g in out["object_mask"]]
        agent_all += a_iou
        object_all += o_iou
        area_all += o_area

        sheet = render_episode_sheet(
            out[obs_col], out["pred_mask"], out["pred_object_mask"],
            out["mask"], out["object_mask"], meta["object_box"],
        )
        sheet_path = os.path.join(out_dir, f"contact_ep{ep_i:02d}.png")
        sheet.save(sheet_path)

        per_ep.append({
            "episode": ep_i, "frames": meta["n_frames"],
            "object_detected": meta["detected"],
            "agent_iou": round(float(np.mean(a_iou)), 4),
            "object_iou": round(float(np.mean(o_iou)), 4),
        })
        print(f"ep {ep_i}: {meta['n_frames']} frames, detected={meta['detected']}, "
              f"agent_iou={np.mean(a_iou):.3f}, object_iou={np.mean(o_iou):.3f} -> {sheet_path}")

    agent_sum = summarize_iou(agent_all)
    object_sum = summarize_iou(object_all, area_all)
    det_rate = detected / len(per_ep) if per_ep else 0.0

    csv_path = os.path.join(out_dir, "iou_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "frames", "object_detected", "agent_iou", "object_iou"])
        for r in per_ep:
            w.writerow([r["episode"], r["frames"], r["object_detected"], r["agent_iou"], r["object_iou"]])
        w.writerow([])
        w.writerow(["OVERALL", f"{len(per_ep)} eps", f"det_rate={det_rate:.3f}", "", ""])
        w.writerow(["agent", "", "", f"mean={agent_sum['mean']:.4f}", f"median={agent_sum['median']:.4f}"])
        w.writerow(["object", "", "", f"mean={object_sum['mean']:.4f}",
                    f"median={object_sum['median']:.4f} area_wtd={object_sum['area_weighted']:.4f}"])

    print("\n=== SUMMARY ===")
    print(f"episodes={len(per_ep)}  frames={len(agent_all)}  object_detection_rate={det_rate:.3f}")
    print(f"agent  IoU: mean={agent_sum['mean']:.4f} median={agent_sum['median']:.4f} "
          f"p10={agent_sum['p10']:.4f} p90={agent_sum['p90']:.4f}")
    print(f"object IoU: mean={object_sum['mean']:.4f} median={object_sum['median']:.4f} "
          f"p10={object_sum['p10']:.4f} p90={object_sum['p90']:.4f} area_wtd={object_sum['area_weighted']:.4f}")
    print(f"artifacts -> {out_dir}")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "push-v3"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    main(task, n)
