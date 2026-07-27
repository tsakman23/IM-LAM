"""IM-LAM Stage-1 overfit gate (Phase 7): the last cheap check before GPU-hours.

Trains IMLAMIDM on a tiny fixed set of transitions (default 64) from one staged task until the
*masked* reconstruction loss collapses toward 0. A correctly-wired, over-parameterized model MUST be
able to memorize 64 frames; if it cannot - in particular if the OBJECT term L_O stays high while the
agent term L_A falls - the interaction FDM (the new, risky machinery) is broken, and you have found out
in minutes instead of seven hours into a full run.

Why this is a valid gate even on distracted frames: the loss is masked to the agent / object regions,
so pixels outside the masks (the distractors) never enter L_A / L_O. Distractor-free vs distracted is
therefore irrelevant to what the gate measures - the staged (distracted) data is used as-is, which also
exercises the real input path.

Data: uses the *test* split by default - it is a fraction of train's size (sweep-into: 4.2GB vs 41GB),
so the one-time load_from_disk (which memory-maps every shard) is minutes not tens of minutes, and for
a memorization check any 64 real transitions serve equally. The fixed batch is loaded ONCE and reused
across both loss arms.

Runs in fp32 (not bf16): the gate asserts the loss reaches a tiny absolute floor, and bf16's ~1e-2
relative precision would stop it short for the wrong reason.

Usage (env per docs/experiments.md):
    conda_env/bin/python scripts/imlam_overfit_gate.py --loss both
    conda_env/bin/python scripts/imlam_overfit_gate.py --loss dual --task sweep-into-v3 --steps 3000

Exit code 0 iff every requested loss passes, so this can gate a launch script.
"""
import argparse
import os
import sys

import numpy as np
import torch
from gymnasium import spaces
from torch.utils.data import DataLoader, Subset

import hydra
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ifo.common.utils.utility import tensordict_collate  # noqa: E402

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "configs"))
CONFIG_BY_LOSS = {"union": "imlam_dmw_stage_1", "dual": "imlam_dual_dmw_stage_1"}


def _compose(loss_name, args):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(
            config_name=CONFIG_BY_LOSS[loss_name],
            overrides=[
                f"env.name=Meta-World/masked-MT1-{args.task}",
                f"dataset.dataset_path={args.data_path}",
                # ++ (force add-or-override): the union config's struct has no such key; the dual config does.
                "++module.log_dual_loss_grad_every=0",  # no trainer -> no global_step for the grad diagnostic
            ],
        )


def _build_fixed_batch(cfg, args, device):
    """Instantiate the real Stage-1 dataset (once) and pull one fixed batch of n_transitions."""
    print(f"loading {args.split} split from {args.data_path} (memory-maps every shard - one-time cost)...", flush=True)
    dataset = hydra.utils.instantiate(cfg.dataset, split=args.split)
    subset = Subset(dataset, list(range(min(args.n_transitions, len(dataset)))))
    loader = DataLoader(subset, batch_size=len(subset), collate_fn=tensordict_collate, shuffle=False)
    batch = next(iter(loader)).to(device)
    print(f"loaded batch: observation={tuple(batch['observation'].shape)}", flush=True)
    return batch


def _build_module(cfg, batch, device):
    """Build IMLAMIDM + SLAPOIDMModule exactly as the experiment does, with spaces derived from data."""
    _, t, c, h, w = batch["observation"].shape
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(t, c, h, w), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(batch["action"].shape[-1],), dtype=np.float32)
    net = hydra.utils.instantiate(cfg.net)(observation_space=obs_space, action_space=act_space)
    module = hydra.utils.instantiate(cfg.module, net=net)
    module.to(device)
    module.train()
    return module


def _run_one_loss(loss_name, cfg, batch, args, device):
    """Overfit the (shared) fixed batch under one loss; return whether it passed."""
    torch.manual_seed(args.seed)
    module = _build_module(cfg, batch, device)
    optimizer = torch.optim.AdamW(module.net.parameters(), lr=args.lr)

    print(f"\n=== overfit gate: loss={loss_name}  task={args.task}  n={batch['observation'].shape[0]}  "
          f"steps={args.steps}  lr={args.lr} ===")
    header = "  step |    L_A    |    L_O    | beta_a_min beta_o_min | proj_a  proj_o"
    keys = ("reconstruction_loss_agent", "reconstruction_loss_object") if loss_name == "dual" \
        else ("reconstruction_loss",)
    last = {}
    for step in range(args.steps + 1):
        step_dict = module.training_step(batch, batch_idx=0, batch_len=1)
        loss = step_dict["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            diag = module.net.interaction_diagnostics()
            last = {k: step_dict[k].item() for k in keys}
            if step % (args.log_every * 10) == 0 or step == args.steps:
                print(header)
            l_a = step_dict.get("reconstruction_loss_agent", step_dict["reconstruction_loss"]).item()
            l_o = step_dict.get("reconstruction_loss_object", step_dict["reconstruction_loss"]).item()
            print(f"  {step:5d} | {l_a:.6f} | {l_o:.6f} |   {diag['beta_msa_a_min'].item():.3f}     "
                  f"{diag['beta_msa_o_min'].item():.3f}  | {diag['proj_a_norm'].item():.4f} "
                  f"{diag['proj_o_norm'].item():.4f}", flush=True)

    passed = all(v < args.threshold for v in last.values())
    print(f"  --> {'PASS' if passed else 'FAIL'} (loss={loss_name}): "
          + ", ".join(f"{k}={v:.6f}" for k, v in last.items()) + f"  [threshold {args.threshold}]")
    if not passed:
        print("      L_O stuck high while proj_o ~ 0 would mean the object branch never engaged; "
              "debug before any full run.")
    return passed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loss", choices=["union", "dual", "both"], default="both",
                   help="Which loss arm(s) to gate. IM-LAM will be run under both, so both is the default.")
    p.add_argument("--task", default="sweep-into-v3", help="Staged Meta-World task (without the Meta-World/ prefix).")
    p.add_argument("--data-path", default="datasets/slapo_local/sweep-into-v3",
                   help="Local save_to_disk dataset dir (load_from_disk).")
    p.add_argument("--split", default="test", help="Split to draw the fixed batch from (test is smaller -> faster load).")
    p.add_argument("--n-transitions", type=int, default=64, help="Number of fixed transitions to overfit.")
    p.add_argument("--steps", type=int, default=3000, help="Optimizer steps.")
    # 3e-4 matches the real Stage-1 lr. Do NOT raise much above this: the dual loss's separately-
    # normalized small-object gradient overshoots and saturates the decoder tanh above ~4e-4 (out_std->0,
    # gradient dies, L_O freezes) - a constant-lr overfit artifact, not present under the real cosine-from-3e-4
    # schedule. If dual ever freezes here, lower the lr or add gradient clipping rather than suspecting the net.
    p.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate (constant, no schedule).")
    p.add_argument("--threshold", type=float, default=1e-2, help="Pass iff each masked loss term falls below this.")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    losses = ["union", "dual"] if args.loss == "both" else [args.loss]

    # The dataset block is identical across the loss configs, so load the fixed batch ONCE and reuse it.
    cfgs = {name: _compose(name, args) for name in losses}
    batch = _build_fixed_batch(cfgs[losses[0]], args, device)

    results = {name: _run_one_loss(name, cfgs[name], batch, args, device) for name in losses}

    print("\n=== overfit gate summary ===")
    for name, ok in results.items():
        print(f"  {name:5s}: {'PASS' if ok else 'FAIL'}")
    all_passed = all(results.values())
    print(f"  overall: {'PASS - safe to launch the full run' if all_passed else 'FAIL - do not launch; debug first'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
