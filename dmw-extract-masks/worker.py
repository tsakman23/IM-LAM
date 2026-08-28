"""Post-hoc mask-extraction worker for the DMW dataset.

Streams the existing dataset, groups frames into episodes from the stored
terminated/truncated flags, detects the object on frame 0, propagates both
entities' masks with SAM2, and writes a superset shard that carries every
original column verbatim plus pred_mask / pred_object_mask.

This module NEVER imports metaworld/gymnasium and never re-simulates: it only
reads existing frames (the no-regeneration invariant).
"""
import io
import logging
import os
from typing import Iterator

import numpy as np
import PIL.Image

logger = logging.getLogger("dmw-extract-masks.worker")


def iter_episodes(rows, term_col: str = "terminated", trunc_col: str = "truncated") -> Iterator[list]:
    """Yield one list of rows per episode. An episode ends on a row whose
    terminated or truncated flag is set; a trailing run with no terminal flag is
    yielded as a final (incomplete) episode."""
    buffer: list = []
    for row in rows:
        buffer.append(row)
        if row.get(term_col) or row.get(trunc_col):
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def decode_frame(value) -> PIL.Image.Image:
    """Decode a stored observation into an RGB PIL image (used only to feed the
    models; pass-through of the original column value is handled separately)."""
    if isinstance(value, PIL.Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("bytes") is not None:
        return PIL.Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, np.ndarray):
        return PIL.Image.fromarray(value.astype(np.uint8)).convert("RGB")
    if isinstance(value, list):
        return PIL.Image.fromarray(np.array(value, dtype=np.uint8)).convert("RGB")
    raise TypeError(f"Cannot decode observation of type {type(value)}")


def extract_episode(episode_rows, detector, predictor, cfg, obs_col, orig_columns):
    """Detect the object on frame 0, propagate both entities, and build the
    output rows: every original column copied verbatim, plus pred_mask and
    pred_object_mask. Returns ``(rows_dict, meta)``."""
    frames = [decode_frame(r[obs_col]) for r in episode_rows]
    object_box, detected = detector.detect_object(
        frames[0], cfg["object_prompt"],
        cfg["box_threshold"], cfg["text_threshold"],
        cfg["fallback_box_threshold"], cfg["fallback_text_threshold"],
    )
    masks = predictor.process_episode(frames, cfg["agent_bbox"], object_box)

    out = {col: [] for col in orig_columns}
    out["pred_mask"] = []
    out["pred_object_mask"] = []
    for i, row in enumerate(episode_rows):
        for col in orig_columns:
            out[col].append(row[col])
        out["pred_mask"].append(masks["agent"][i])
        out["pred_object_mask"].append(masks["object"][i])

    meta = {"detected": detected, "object_box": object_box, "n_frames": len(frames)}
    return out, meta


def write_shard(rows_dict: dict, shard_path: str, image_columns) -> None:
    """Write a superset shard to parquet, encoding the given columns as images."""
    from datasets import Dataset
    from datasets import Image as ImageFeature

    ds = Dataset.from_dict(rows_dict)
    for col in image_columns:
        if col in ds.column_names:
            ds = ds.cast_column(col, ImageFeature())
    ds.to_parquet(shard_path)


def _open_source(config, cfg, task, split, source):
    """Return a streaming iterable of rows for (task, split). ``source`` may be a
    list/glob of local parquet files; otherwise the configured hub dataset is
    streamed. Returns ``(dataset, orig_columns, image_columns)``."""
    from datasets import Image as ImageFeature
    from datasets import load_dataset

    if source is not None:
        ds = load_dataset("parquet", data_files=source, split="train", streaming=True)
    else:
        ds = load_dataset(cfg["source_dataset_id"], name=task, split=split, streaming=True)

    features = ds.features
    orig_columns = list(features.keys())
    image_columns = [c for c, f in features.items() if isinstance(f, ImageFeature)]
    return ds, orig_columns, image_columns


def run_worker(task, split, config, device="cuda:0", source=None, max_episodes=None):
    """Stream (task, split), extract masks per episode, write superset shards.

    Returns stats: ``{"episodes", "detected", "detection_rate", "frames", "shards"}``.
    """
    from detector import GroundingDinoDetector
    from episode import Sam2DualPredictor
    from utils import get_torch_dtype, resolve_task_config

    cfg = resolve_task_config(config, task)
    obs_col = cfg["observation_column"]
    ds, orig_columns, image_columns = _open_source(config, cfg, task, split, source)

    detector = GroundingDinoDetector(
        cfg["grounding_dino_model_id"], device,
        object_region=cfg.get("object_region"),
        object_max_box_area=cfg.get("object_max_box_area"),
    )
    predictor = Sam2DualPredictor(
        cfg["sam2_model_id"], device, get_torch_dtype(cfg.get("dtype", "bfloat16"))
    )

    shard_dir = os.path.join(config["output_dir"], task, split)
    os.makedirs(shard_dir, exist_ok=True)
    out_image_columns = image_columns + ["pred_mask", "pred_object_mask"]
    save_interval = config.get("save_interval", 500)

    accum = {c: [] for c in orig_columns + ["pred_mask", "pred_object_mask"]}
    episodes = detected = frames = shard_idx = 0
    shards = []

    def flush():
        nonlocal shard_idx, accum
        if not accum["pred_mask"]:
            return
        path = os.path.join(shard_dir, f"shard_{shard_idx:06d}.parquet")
        write_shard(accum, path, out_image_columns)
        shards.append(path)
        logger.info("wrote %s (%d rows)", path, len(accum["pred_mask"]))
        shard_idx += 1
        accum = {c: [] for c in accum}

    for episode_rows in iter_episodes(ds, config.get("terminated_column", "terminated"),
                                      config.get("truncated_column", "truncated")):
        out, meta = extract_episode(episode_rows, detector, predictor, cfg, obs_col, orig_columns)
        for col in accum:
            accum[col].extend(out[col])
        episodes += 1
        detected += int(meta["detected"])
        frames += meta["n_frames"]
        logger.info("episode %d: %d frames, object detected=%s", episodes, meta["n_frames"], meta["detected"])
        if episodes % save_interval == 0:
            flush()
        if max_episodes is not None and episodes >= max_episodes:
            break

    flush()
    return {
        "episodes": episodes,
        "detected": detected,
        "detection_rate": (detected / episodes) if episodes else 0.0,
        "frames": frames,
        "shards": shards,
    }
