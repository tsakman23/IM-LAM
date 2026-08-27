"""Run the three IM-LAM Stage-1 mechanism diagnostics on a frozen checkpoint.

  1. Held-out object-prediction error (S7.4.1): E_O and the copy baseline E_O^copy over the object mask,
     reported as ObjectPredictionRatio = E_O / (E_O^copy + eps). < 1 means the model beats copy.
  2. Object-dynamics probe (S7.4.2): a frozen linear probe from the object read-out O_t to the k-step
     object position delta Delta s^O; reports R²/NMSE, and SeparationRatio = R²(A_t)/R²(O_t) from the
     agent read-out A_t (low = the object read-out is the better predictor of object motion).
  3. Agent-path dependence (S7.4.3): E_O under agent_ctx_mode normal / no_transition / shuffled, reported
     as R_no-transition = E_O^{no-transition}/E_O^{normal} and R_shuffled (forward-time, no retraining).

The per-batch cores live in this file (tested on synthetic data in tests/test_object_diagnostics.py via
metrics.py + the model's extract_entities); this script only adds checkpoint/data loading and aggregation.

Every run also appends one row to results/imlam_diagnostics.csv by default (fixed header, safe to
accumulate across models/tasks/seeds - see --results-csv/--run-name/--run-seed). Run
results_table.py afterward to pivot that CSV into a readable Markdown summary.

--loss, --run-seed, and --wandb-run-id's run id are all auto-derived from the checkpoint path by
default (its parent directory name, e.g. 'im-lam_handle-pull_dual_seed2-1'): 'dual' if that
word appears else 'union'; the digits after 'seed'; and the dir name with the trailing pipeline-stage
suffix (-1/-2/-3) stripped, respectively. Pass --loss/--run-seed explicitly to override.

Usage (env per docs/experiments.md):
    python scripts/imlam_diagnostics/run_diagnostics.py \
        --checkpoint checkpoints/<imlam_run>/step-000031248.ckpt --task handle-pull-v3 --model imlam --wandb-run-id
"""
import argparse
import csv
import datetime
import os
import re
import sys
from typing import Optional

import numpy as np
import torch
from gymnasium import spaces
from torch.utils.data import DataLoader

import hydra
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ifo.common.utils.utility import tensordict_collate  # noqa: E402
from ifo.modules.slapo.experiment import SLAPOExperiment  # noqa: E402
from scripts.imlam_diagnostics.metrics import (  # noqa: E402
    EPS,
    fit_linear_probe,
    masked_sq_error_and_area,
    probe_scores,
    quat_geodesic_angle,
    separation_ratio,
)

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "configs"))
# (model, loss) -> Stage-1 config. IM-LAM and Foreground share the diagnostics; the direct-z ablation
# uses --config-name imlam_direct_z_dmw_stage_1.
CONFIG_BY_MODEL_LOSS = {
    ("imlam", "union"): "imlam_dmw_stage_1",
    ("imlam", "dual"): "imlam_dual_dmw_stage_1",
    ("foreground", "union"): "foreground_masklam_dmw_stage_1",
    ("foreground", "dual"): "foreground_masklam_dual_dmw_stage_1",
}

DEFAULT_RESULTS_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "imlam_diagnostics.csv"))
# Fixed superset of columns: not every model produces every metric (R_no_transition/R_shuffled are
# IM-LAM-only, since Foreground/direct-z have no agent_ctx_mode), so the header stays stable across
# appends regardless of which run wrote it - missing values are just blank cells, not missing columns.
RUN_FIELDS = ["timestamp", "run_name", "seed", "model", "loss", "task", "checkpoint", "config_name",
              "split", "batch_size", "max_batches"]
METRIC_FIELDS = ["E_O", "E_O_copy", "ObjectPredictionRatio", "R_no_transition", "R_shuffled",
                  "R2_object", "NMSE_object", "Var_delta", "R2_agent", "SeparationRatio"]
CSV_FIELDNAMES = RUN_FIELDS + METRIC_FIELDS


def _checkpoint_dirname(checkpoint: str) -> str:
    """The checkpoint's parent directory name, e.g. 'im-lam_handle-pull_union_seed2-1' from
    '.../checkpoints/im-lam_handle-pull_union_seed2-1/step-000031248.ckpt'."""
    return os.path.basename(os.path.dirname(os.path.abspath(checkpoint)))


def infer_wandb_run_id(checkpoint: str) -> str:
    """Strip the trailing pipeline-stage suffix from the checkpoint dir name: the '-1'/'-2'/'-3'
    dirs are Stage 1/2/3 checkpoints of ONE W&B run, not separate runs - e.g.
    'im-lam_handle-pull_union_seed2-1' -> 'im-lam_handle-pull_union_seed2'."""
    return re.sub(r"-\d+$", "", _checkpoint_dirname(checkpoint))


def infer_seed(checkpoint: str) -> Optional[str]:
    """The digits following 'seed' in the checkpoint dir name, e.g. '...seed2-1' -> '2'."""
    m = re.search(r"seed(\d+)", _checkpoint_dirname(checkpoint), re.IGNORECASE)
    return m.group(1) if m else None


def infer_loss(checkpoint: str) -> str:
    """'dual' if that word appears (case-insensitive) in the checkpoint dir name, else 'union'."""
    return "dual" if "dual" in _checkpoint_dirname(checkpoint).lower() else "union"


def append_results_csv(csv_path, run_info, flat_results):
    """Append one row (this run's metadata + metrics) to the accumulating results CSV.

    Append-only, fixed header (CSV_FIELDNAMES) - repeated runs across models/tasks/seeds build up one
    growing table rather than overwriting each other. Use results_table.py to pivot it into a summary.
    """
    row = {**run_info, **flat_results}
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, restval="", extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nappended results row to {csv_path}")


def push_to_wandb(flat_results, run_id, project, entity, scope="eval"):
    """Write the diagnostics into an EXISTING (already-finished) W&B run's summary via the API.

    No new run, no resume, no history-step juggling: the values appear as summary columns on the original
    training run, directly comparable across the IM-LAM / Foreground / direct-z runs in the W&B runs table.
    Run each model's diagnostics with its own --wandb-run-id after training, and W&B does the comparison.
    """
    import wandb
    path = f"{entity + '/' if entity else ''}{project}/{run_id}"
    run = wandb.Api().run(path)
    run.summary.update({f"{scope}/{key}": value for key, value in flat_results.items()})
    print(f"\npushed {len(flat_results)} diagnostics to W&B run {path} (summary, under {scope}/)")


def load_frozen_net(cfg, checkpoint, obs_space, act_space, device):
    """Build IMLAMIDM from cfg and load the trained net.* weights from a Stage-1 checkpoint."""
    net = hydra.utils.instantiate(cfg.net)(observation_space=obs_space, action_space=act_space)
    if checkpoint is not None:
        model_sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        exp = object.__new__(SLAPOExperiment)
        net.load_state_dict(exp.filter_state_dict_by_prefix(model_sd, "net"), strict=True)
    return net.to(device).eval()


# ---- per-batch cores (pure given a net + batch; the object of the synthetic smoke test) ----

def _is_imlam(net):
    """IM-LAM marker: has the directed agent-path (agent_ctx_mode) and the interaction read-outs."""
    return hasattr(net, "extract_entities")


def _iter_capped(loader, max_batches, device):
    """Iterate a DataLoader LAZILY (never materialize the whole split - that OOMs on the full test set)
    up to ``max_batches`` batches, moving each to ``device``."""
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        yield batch.to(device)


def batch_probe_data(net, batch, k):
    """Pooled (A_t, O_t) features and the k-step object-motion delta ``[pos(3), rotation-angle(1)]``.

    Uses ``net.probe_features`` (uniform across model types). The target is the Euclidean position delta
    plus the quaternion geodesic angle, so a rotation-dominant object is not invisible to the probe.
    """
    fs = net.frame_stack
    with torch.no_grad():
        a_feat, o_feat = net.probe_features(batch["observation"], batch["mask"], batch["object_mask"])
    state = batch["object_state"]  # (B, T, 7) = [pos(3), quat(4)]
    pos_delta = state[:, fs - 1 + k, :3] - state[:, fs - 1, :3]                              # (B, 3)
    angle = quat_geodesic_angle(state[:, fs - 1, 3:], state[:, fs - 1 + k, 3:]).unsqueeze(-1)  # (B, 1)
    return a_feat, o_feat, torch.cat([pos_delta, angle], dim=-1)                             # (B, 4)


# ---- aggregation (lazy iteration; no full-split materialization) ----

def object_error_and_agent_path(net, loader, device, max_batches):
    """Diagnostics 1 (S7.4.1) + 3 (S7.4.3), one pass over a SHUFFLED loader.

    Pooled S7.4.1 estimator: accumulate summed masked error and summed mask area across batches, then
    divide once - weighting object pixels equally, per the proposal, not a batch-mean of normalized errors.
    The loader MUST be shuffled: the ``shuffled`` agent_ctx_mode rolls the predicted-agent context by one
    within the batch, so a sequential loader would land on the adjacent (near-identical) frame and
    under-report dependence. Agent-path is IM-LAM only (baselines have no predicted agent transition).
    """
    is_imlam = _is_imlam(net)
    modes = ("normal", "no_transition", "shuffled") if is_imlam else ("normal",)
    err_sum = {m: torch.zeros((), device=device) for m in modes}
    copy_sum = torch.zeros((), device=device)
    area_sum = torch.zeros((), device=device)
    for batch in _iter_capped(loader, max_batches, device):
        fs = net.frame_stack
        target = batch["observation"][:, fs]
        current = batch["observation"][:, fs - 1]
        om = batch["object_mask"][:, fs]
        for m in modes:
            kwargs = {"agent_ctx_mode": m} if is_imlam else {}
            with torch.no_grad():
                pred = net(batch["observation"], batch["mask"], object_mask=batch["object_mask"], **kwargs)[0]
            err, area = masked_sq_error_and_area(pred, target, om)
            err_sum[m] = err_sum[m] + err
            if m == "normal":
                copy_err, _ = masked_sq_error_and_area(current, target, om)
                copy_sum = copy_sum + copy_err
                area_sum = area_sum + area
    e_o = err_sum["normal"] / area_sum.clamp(min=1.0)
    e_copy = copy_sum / area_sum.clamp(min=1.0)
    out = {"E_O": e_o, "E_O_copy": e_copy, "ObjectPredictionRatio": e_o / (e_copy + EPS)}
    for m in modes:
        if m != "normal":
            out[f"R_{m}"] = err_sum[m] / (err_sum["normal"] + EPS)  # area cancels
    return out


def object_probe_diagnostic(net, loader, device, k, max_batches, train_frac=0.7):
    """Diagnostic 2: the object-dynamics probe (S7.4.2), over a SEQUENTIAL loader.

    Accumulates the pooled (A_t, O_t) features (small - not the raw frames) and the k-step object-motion
    delta, then fits a linear probe on the first ``train_frac`` (temporally contiguous - no leakage) and
    scores it on the rest. Two probes (from O_t and A_t) give R²/NMSE and the SeparationRatio (low = O_t
    predicts object motion far better than A_t). Needs the whole split at once -> post-run, not per-batch.
    """
    a_feats, o_feats, deltas = [], [], []
    for batch in _iter_capped(loader, max_batches, device):
        a, o, d = batch_probe_data(net, batch, k)
        a_feats.append(a); o_feats.append(o); deltas.append(d)
    A, O, y = torch.cat(a_feats), torch.cat(o_feats), torch.cat(deltas)
    n_tr = int(train_frac * len(y))

    def score(feats):  # fit on the train split, score on the held-out split
        w = fit_linear_probe(feats[:n_tr], y[:n_tr])
        return probe_scores(feats[n_tr:], y[n_tr:], w)

    s_o, s_a = score(O), score(A)
    return {"R2_object": s_o["r2"], "NMSE_object": s_o["nmse"], "Var_delta": s_o["var"],
            "R2_agent": s_a["r2"], "SeparationRatio": separation_ratio(s_a["r2"], s_o["r2"])}


def _loader(dataset, batch_size, shuffle, seed):
    """A DataLoader (seeded when shuffled) - not materialized; the caller iterates it lazily."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, collate_fn=tensordict_collate,
                      shuffle=shuffle, drop_last=True, generator=generator)


def main():
    """Load a frozen Stage-1 checkpoint + a held-out split, run all three diagnostics, print a report.

    Kept standalone (rather than folded into training) because its headline output is a CROSS-model
    comparison - the same metrics for IM-LAM, the same-loss Foreground baseline, and the direct-z ablation
    - inherently post-hoc across separate runs on a common held-out split, pushed onto each run's W&B
    summary. (Per-run, the per-batch diagnostics 1 and 3 also log during Stage-1 validation.)
    """
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Frozen Stage-1 checkpoint (.ckpt).")
    p.add_argument("--model", choices=["imlam", "foreground"], default="imlam",
                   help="Which model the checkpoint is (agent-path is auto-skipped for foreground).")
    p.add_argument("--loss", choices=["union", "dual"], default=None,
                   help="Defaults to auto-detected from the checkpoint path: 'dual' if that word "
                         "appears in the checkpoint's directory name, else 'union'.")
    p.add_argument("--config-name", default=None,
                   help="Override the composed config, e.g. imlam_direct_z_dmw_stage_1 for the ablation.")
    p.add_argument("--task", default=None, required=True, help="DMW task name (e.g. handle-pull-v3).")
    p.add_argument("--data-path", default=None, help="Local dataset dir; default = the config's HF repo.")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=1562,
                   help="Batches per diagnostic (loaders iterate lazily). None = whole split (can be huge).")
    p.add_argument("--k", type=int, default=10, help="Probe horizon (matched to future_obs_offset).")
    p.add_argument("--seed", type=int, default=0, help="Seed for the shuffled loader (diagnostics 1 + 3).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb-run-id", action="store_true",
                   help="If set, push the diagnostics into the checkpoint's own W&B run's summary "
                        "(under eval/). The run id is auto-derived from the checkpoint path: its "
                        "directory name with the trailing pipeline-stage suffix (-1/-2/-3) stripped.")
    p.add_argument("--wandb-project", default="masklam")
    p.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV,
                   help="Append this run's results as one row here (fixed header, safe to accumulate "
                        "across models/tasks/seeds). Set to '' to skip writing.")
    p.add_argument("--run-name", default=None,
                   help="Label for this row (default: the checkpoint's parent directory name, e.g. "
                         "'fg_masklam_handle-pull_seed1-1').")
    p.add_argument("--run-seed", default=None,
                   help="Training-seed label for this row (e.g. '1'). Defaults to auto-extracted "
                         "from the checkpoint path (the digits following 'seed'); pass explicitly to "
                         "override. Distinct from --seed, which seeds the diagnostics' own shuffled "
                         "loader.")
    args = p.parse_args()
    device = torch.device(args.device)

    if args.loss is None:
        args.loss = infer_loss(args.checkpoint)
    if args.run_seed is None:
        args.run_seed = infer_seed(args.checkpoint)
    wandb_run_id = infer_wandb_run_id(args.checkpoint) if args.wandb_run_id else None

    # Compose the same Stage-1 config the run used, so the net/dataset are built identically. The grad
    # diagnostic is disabled (no trainer here, so no global_step); data-path override picks a local split.
    overrides = [f"env.name=Meta-World/masked-MT1-{args.task}", "++module.log_dual_loss_grad_every=0"]
    if args.data_path:
        overrides.append(f"dataset.dataset_path={args.data_path}")
    config_name = args.config_name or CONFIG_BY_MODEL_LOSS[(args.model, args.loss)]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=config_name, overrides=overrides)

    dataset = hydra.utils.instantiate(cfg.dataset, split=args.split)
    # Diagnostics 1 + 3 need a SHUFFLED loader (so the 'shuffled' agent-context roll lands on an unrelated
    # sample); the probe needs a SEQUENTIAL loader (temporally contiguous train/eval split, no leakage).
    shuffled_loader = _loader(dataset, args.batch_size, shuffle=True, seed=args.seed)
    sequential_loader = _loader(dataset, args.batch_size, shuffle=False, seed=args.seed)

    # Derive the observation/action spaces from one real batch (avoids instantiating the Meta-World env),
    # then load the frozen trained net.
    sample = next(iter(sequential_loader))
    _, t, c, h, w = sample["observation"].shape
    obs_space = spaces.Box(-np.inf, np.inf, (t, c, h, w), np.float32)
    act_space = spaces.Box(-1.0, 1.0, (sample["action"].shape[-1],), np.float32)
    net = load_frozen_net(cfg, args.checkpoint, obs_space, act_space, device)

    print(f"=== Stage-1 diagnostics: {args.model} / {args.task} ({args.loss}), <= {args.max_batches} batches ===")
    flat_results = {}
    for name, result in (
        ("Object prediction error + agent-path",
         object_error_and_agent_path(net, shuffled_loader, device, args.max_batches)),
        ("Object-dynamics probe",
         object_probe_diagnostic(net, sequential_loader, device, args.k, args.max_batches)),
    ):
        print(f"\n[{name}]")
        for key, value in result.items():
            print(f"  {key:24s} = {value.item():.6f}")
            flat_results[key] = value.item()

    # Push into the training run's summary so all three models' numbers sit together in the W&B runs table.
    if wandb_run_id:
        # if run id isn't found, raise an error (don't silently create a new run - the point is to push into an existing run's summary)
        try:
            push_to_wandb(flat_results, wandb_run_id, args.wandb_project, "gtsakoumakis-newcastle-university")
        except Exception as e:
            print(f"Error pushing to W&B run {wandb_run_id}: {e}")
            sys.exit(1)

    if args.results_csv:
        run_name = args.run_name or os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
        run_info = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "run_name": run_name, "seed": args.run_seed or "", "model": args.model, "loss": args.loss,
            "task": args.task, "checkpoint": args.checkpoint, "config_name": config_name,
            "split": args.split, "batch_size": args.batch_size, "max_batches": args.max_batches,
        }
        append_results_csv(args.results_csv, run_info, flat_results)


if __name__ == "__main__":
    main()
