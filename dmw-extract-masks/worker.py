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
import queue
import threading
from typing import Iterator

import numpy as np
import PIL.Image

logger = logging.getLogger("dmw-extract-masks.worker")


def prefetch_iter(iterable, buffer_size: int = 4):
    """Yield items from ``iterable`` produced on a background thread, so the
    consumer's GPU work overlaps the producer's I/O (streaming + image decode).
    A bounded queue applies backpressure; producer exceptions are re-raised in
    the consumer. This is the key fix for an I/O-starved GPU: the extractor's
    GPU is otherwise idle while each episode is fetched and decoded serially."""
    q: queue.Queue = queue.Queue(maxsize=buffer_size)
    done = object()
    err: list = []

    def _produce():
        try:
            for item in iterable:
                q.put(item)
        except Exception as exc:  # surface to the consumer instead of dying silently
            err.append(exc)
        finally:
            q.put(done)

    thread = threading.Thread(target=_produce, daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is done:
            break
        yield item
    thread.join()
    if err:
        raise err[0]


def partition_shards(shards: list, rank: int, world_size: int) -> list:
    """Round-robin slice of ``shards`` for worker ``rank`` of ``world_size``.
    Lets one process per GPU cover a disjoint, balanced set of shards."""
    return shards[rank::world_size]


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
    """Return ``(dataset, orig_columns, image_columns, num_frames)`` for (task, split).

    ``source`` may be:
      - None                -> stream the configured hub dataset;
      - a directory path    -> a staged ``save_to_disk`` copy (load_from_disk, local, fast);
      - a list of parquet files -> stream those parquet files.
    ``num_frames`` (for the progress-bar ETA) is exact for local sources and
    best-effort (metadata) for the hub.
    """
    from datasets import Image as ImageFeature

    if source is None:
        from datasets import load_dataset
        ds = load_dataset(cfg["source_dataset_id"], name=task, split=split, streaming=True)
        try:
            from datasets import load_dataset_builder
            num_frames = load_dataset_builder(cfg["source_dataset_id"], name=task).info.splits[split].num_examples
        except Exception:
            num_frames = None
    elif isinstance(source, str):
        from datasets import load_from_disk
        ds = load_from_disk(source)
        num_frames = len(ds)
    else:
        import pyarrow.parquet as pq
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=source, split="train", streaming=True)
        num_frames = sum(pq.ParquetFile(f).metadata.num_rows for f in source)

    features = ds.features
    image_columns = [c for c, f in features.items() if isinstance(f, ImageFeature)]
    return ds, list(features.keys()), image_columns, num_frames


def run_worker(task, split, config, device="cuda:0", source=None, max_episodes=None):
    """Stream (task, split), extract masks per episode, write superset shards.

    Returns stats: ``{"episodes", "detected", "detection_rate", "frames", "shards"}``.
    """
    from detector import GroundingDinoDetector
    from episode import Sam2DualPredictor
    from utils import get_torch_dtype, resolve_task_config

    cfg = resolve_task_config(config, task)
    obs_col = cfg["observation_column"]
    ds, orig_columns, image_columns, total_frames = _open_source(config, cfg, task, split, source)

    detector = GroundingDinoDetector(
        cfg["grounding_dino_model_id"], device,
        object_region=cfg.get("object_region"),
        object_max_box_area=cfg.get("object_max_box_area"),
        object_box_offset=cfg.get("object_box_offset"),
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

    from tqdm.auto import tqdm

    episode_stream = iter_episodes(ds, config.get("terminated_column", "terminated"),
                                   config.get("truncated_column", "truncated"))
    # Prefetch overlaps streaming+decode (producer thread) with GPU inference
    # (this loop), so the otherwise-idle GPU stays fed.
    pbar = tqdm(total=total_frames, unit="frame", desc=f"{task}/{split}", dynamic_ncols=True)
    for episode_rows in prefetch_iter(episode_stream, buffer_size=config.get("prefetch_buffer", 4)):
        out, meta = extract_episode(episode_rows, detector, predictor, cfg, obs_col, orig_columns)
        for col in accum:
            accum[col].extend(out[col])
        episodes += 1
        detected += int(meta["detected"])
        frames += meta["n_frames"]
        # Log only failures; successes are summarized by the progress bar.
        if not meta["detected"]:
            logger.warning("%s/%s episode %d: object NOT detected (empty object mask)",
                           task, split, episodes)
        pbar.update(meta["n_frames"])
        pbar.set_postfix(ep=episodes, det=f"{detected}/{episodes}", refresh=False)
        if episodes % save_interval == 0:
            flush()
        if max_episodes is not None and episodes >= max_episodes:
            break

    pbar.close()
    flush()
    return {
        "episodes": episodes,
        "detected": detected,
        "detection_rate": (detected / episodes) if episodes else 0.0,
        "frames": frames,
        "shards": shards,
    }
