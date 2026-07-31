# Experiment commands

End-to-end reference in the actual order to run things:
**generate (only for a new task) -> stage -> run -> visualize.**

Replace `<task>` with a task slug, e.g. `push-v3`, `push-wall-v3`, `plate-slide-v3`,
`door-open-v3`, `faucet-open-v3`, `handle-pull-v3`, `dial-turn-v3`, `pick-place-v3`,
`peg-insert-side-v3`, `sweep-into-v3`.

Two datasets are involved:
- **`EpicPinkPenguin/visual_distracting_metaworld`** - the MaskLAM authors' release (agent mask only). Used by **MaskLAM**.
- **`tsakman23/visual_masked_distracting_metaworld`** - my regenerated data (adds `object_mask` + `object_state`). Used by **Foreground-MaskLAM** and **IM-LAM**.

---

## 0. Environment (prerequisites for everything below)

- Activate the conda env: `conda activate ./conda_env`.
- Always export:
  ```bash
  export MUJOCO_GL=egl                              # headless EGL rendering
  export HF_HOME=/data2/masklam/datasets/hf_home    # keep HF caches off the home dir (quota)
  ```
- HuggingFace auth (needed to push, and to pull data): add a **write** token as `HF_TOKEN` in
  `~/.bashrc`. Confirm with `hf auth whoami` -> should print your username.

---

## 1. Data generation (your own object-mask data) - only for a NEW task

Skip this if the task is already on `tsakman23/...`. Object masks are ground-truth simulator masks emitted *by the
generator itself* (`object_mask` + `object_state`) - there is no separate mask step, and SAM is not
supported (yet). `--push_to_hub` uploads to `tsakman23/...` (config = task slug).

```bash
cd distracting-metaworld-dataset-main
# train: 1M steps, seed 0
python generate_dataset_huggingface.py \
  --env Meta-World/MT1-<task> --split train --num_steps 1000000 --seed 0 --push_to_hub
# test: 100k steps, seed 100000  (can run in parallel)
python generate_dataset_huggingface.py \
  --env Meta-World/MT1-<task> --split test --num_steps 100000 --seed 100000 --push_to_hub
```
- ~5 h per 1M-step split. Without `--push_to_hub` the data stays only in the scratch cache
  `datasets/hf_cache/<task>_<split>`.
- `--save_local [DIR]` also writes a training-loadable `save_to_disk` copy while generating
  (combinable with `--push_to_hub`). The generator runs from a subdir, so pass an **absolute** DIR,
  e.g. `--save_local /data2/masklam/datasets/slapo_local` - then you can skip Section 2 for this task.
- Articulated objects (door/faucet/handle/dial): if the object mask looks wrong, add the task to
  `TASK_OBJECT_OVERRIDES` in `generate_dataset_huggingface.py`; `push`/`door-open` use the default heuristic.

**Re-push from cache** if a `--push_to_hub` upload failed (e.g. auth) - no regeneration:
```bash
cd distracting-metaworld-dataset-main
python verify/repush_from_cache.py --config <task> --split train   # ...and --split test
```

---

## 2. Data staging (make a fast, training-ready local copy) - DO THIS BEFORE EACH RUN

Training reads the dataset every step. On `/data2` (a single spinning disk) several concurrent runs
saturate it and stall (see Section 6). So stage a `save_to_disk` copy to **RAM (`/tmp`)** once; then
training reads it via `load_from_disk` - **no build, no fingerprint stubs, RAM-speed reads.** The
command downloads from HF and/or reuses the built cache automatically, and is **idempotent** (skips a
split already present). Run it from the repo root.

```bash
# MaskLAM data (authors' release):
python scripts/save_local_dataset.py \
  --repo EpicPinkPenguin/visual_distracting_metaworld --task <task> \
  --splits test train --out-root /tmp/slapo_local

# Foreground-MaskLAM data (object masks):
python scripts/save_local_dataset.py \
  --repo tsakman23/visual_masked_distracting_metaworld --task <task> \
  --splits test train --out-root /tmp/slapo_local
```
Every run then adds `dataset.dataset_path=/tmp/slapo_local` (Section 5). Both dataset types stage the
same way - only `--repo` differs.

- Stage only the tasks you're about to run; `rm -rf /tmp/slapo_local/<task>` before the next batch.
  **`/tmp` is wiped on reboot** - re-run this step afterwards (fast, reuses the on-disk build).
- Run this **serially per task** (not many at once). Seeds of the same task share one copy.
- **Persistent alternative:** `--out-root ./datasets/slapo_local` (on `/data2`, survives reboots).
  Reads are then disk-bound, so start ~2 runs at a time to let the page cache warm (Section 6).
- **One-shot acquire + stage:** `python scripts/download_dataset.py --env Meta-World/masked-MT1-<task>
  [--repo tsakman23/... --with-object-mask] --save-local /tmp/slapo_local` does fetch + local copy in
  one call. Plain `download_dataset.py` (no `--save-local`) instead pre-builds the persistent
  `./datasets/slapo/<env>` HF cache used by the repo-id load path.

---

## 3. Verify a dataset (optional)

```bash
cd distracting-metaworld-dataset-main
# IID vs the authors' release (checks first 3k rows; raise --n for more)
python verify/iid_check.py --config <task> --split train --n 3000
# Visual mask panels (observation | agent mask | object mask | overlay)
python verify/render_mask_panels.py \
  --env Meta-World/MT1-<task> --steps 150 --num_frames 8 --out datasets/verify/mask_panels/<task>
```

---

## 4. (context) How the run reads the dataset

The full pipeline runs Stage 1 (IDM/FDM) -> Stage 2 (latent policy) -> Stage 3 (BC) + rollout eval in
one process / one W&B run. Two load modes:

- **Local (recommended, `dataset.dataset_path=<DIR>`):** `load_from_disk` on the Section-2 copy at
  `<DIR>/<task>/<split>` - no build, no fingerprint stubs; `cache_dir` is ignored.
- **Repo-id (default, no override):** `load_dataset` from HF, building a parquet->Arrow cache in
  `dataset.cache_dir` (defaults to `./datasets/slapo/${env.name}`). First run builds; later runs
  cache-hit. Prone to the disk jam if several runs build/read at once (Section 6).

---

## 5. Run experiments

Both commands are identical except the Stage-1 config and `--repo` you staged in Section 2.

### 5a. MaskLAM (baseline)
```bash
CUDA_VISIBLE_DEVICES=<n> MUJOCO_GL=egl python experiments/run_slapo.py \
  run_id=masklam_<task>_seed1 env.name=Meta-World/masked-MT1-<task> \
  dataset.dataset_path=/tmp/slapo_local \
  logger.mode=online logger.group=masklam_reprod logger.notes="MaskLAM <task> seed 1" \
  trainer.compile=True fabric.precision=bf16-mixed trainer.random_seed=1 \
  --stage stage_1 -cn slapo_dmw_stage_1 \
  --stage stage_2 -cn slapo_dmw_stage_2 \
  --stage stage_3 -cn slapo_dmw_stage_3
```
- Uses the authors' `EpicPinkPenguin` release (agent mask only), pinned in `slapo_dmw_stage_1/2/3.yaml`
  with `with_object_mask=false`. Stage it with `--repo EpicPinkPenguin/...` in Section 2.

### 5b. Foreground-MaskLAM (baseline)
```bash
CUDA_VISIBLE_DEVICES=<n> MUJOCO_GL=egl python experiments/run_slapo.py \
  run_id=fg_masklam_<task>_seed1 env.name=Meta-World/masked-MT1-<task> \
  dataset.dataset_path=/tmp/slapo_local \
  logger.mode=online logger.group=foreground_masklam logger.notes="Foreground-MaskLAM <task> seed1" \
  trainer.compile=True fabric.precision=bf16-mixed trainer.random_seed=1 \
  --stage stage_1 -cn foreground_masklam_dmw_stage_1 \
  --stage stage_2 -cn foreground_masklam_dmw_stage_2 \
  --stage stage_3 -cn foreground_masklam_dmw_stage_3
```
- Stage 1 uses `foreground_masklam_dmw_stage_1` (object-mask union loss, `tsakman23` data). Stage it
  with `--repo tsakman23/...` in Section 2. Stages 2/3 use `foreground_masklam_dmw_stage_2/3`, which
  are the stock stage 2/3 with the dataset pinned to the same `tsakman23` repo - so the whole arm
  runs on one dataset and no cross-dataset (authors' release vs regenerated) caveat enters the
  comparison. Only `tsakman23` needs staging for this arm. (Stages 2/3 never read object masks; in
  the staged-local workflow the `dataset.dataset_path=/tmp/slapo_local` override already fed all
  stages the staged tsakman23 copy - the pinned configs make the repo-id fallback match.)
- **Loss-only** design: `object_mask_loss=true`, `object_mask_input=false`. The agent ∪ object union
  gates the FDM reconstruction loss (decoder must reconstruct the object), while the encoder/IDM mask
  input stays agent-only - so `z_t` remains an embodiment action and the frozen encoder stays
  consistent with stages 2/3 (which feed it the agent mask).
- `object_mask_input=true` is an opt-in experiment (feed the union into the encoder too); it changes
  what `z_t` encodes and would need the union threaded through stages 2/3 - don't enable without that.
- Object-mask tasks on `tsakman23/...`: `push-v3`, `door-open-v3`, `sweep-into-v3`, `handle-pull-v3`,
  `pick-place-v3`, `dial-turn-v3`, `peg-insert-side-v3` all have **both** splits now (runnable).

### 5c. Dual-loss variant (baseline)
```bash
CUDA_VISIBLE_DEVICES=<n> MUJOCO_GL=egl python experiments/run_slapo.py \
  run_id=dual_masklam_<task>_seed1 env.name=Meta-World/masked-MT1-<task> \
  dataset.dataset_path=/tmp/slapo_local \
  logger.mode=online logger.group=dual_masklam logger.notes="FG Dual-loss <task> seed1" \
  trainer.compile=True fabric.precision=bf16-mixed trainer.random_seed=1 \
  --stage stage_1 -cn foreground_masklam_dual_dmw_stage_1 module.object_loss_weight=1.0 \
  --stage stage_2 -cn foreground_masklam_dmw_stage_2 \
  --stage stage_3 -cn foreground_masklam_dmw_stage_3
```
- Stage 1 uses `foreground_masklam_dual_dmw_stage_1` (same `tsakman23` object-mask data as
  Foreground-MaskLAM, §5b). Stage it with `--repo tsakman23/...` in Section 2. Stages 2/3 use
  `foreground_masklam_dmw_stage_2/3` (stock stage 2/3 pinned to `tsakman23`), same as 5b.
- **Loss-only** design, same split as Foreground-MaskLAM: `object_dual_loss=true`,
  `object_mask_loss=false`, `object_mask_input=false` - agent-only encoder/IDM input, so `z_t` and
  stages 2/3 stay consistent with 5a/5b.
- The difference from 5b: `L = L_A + object_loss_weight * L_O`, where `L_A` and `L_O` are each
  normalized by their **own** mask area, instead of one union mask sharing a single normalization. A
  small object no longer competes with the much larger agent mask for weight - it gets its own budget,
  sized by `object_loss_weight` (`lambda_O`) alone. `object_mask_loss` and `object_dual_loss` are
  mutually exclusive (raises at module init if both are set).
- Sweep the object weight from the CLI, no yaml edit needed: `module.object_loss_weight=<value>` on the
  Stage-1 line (default `1.0`).
- Per-term losses are logged separately to W&B - `reconstruction_loss_agent` / `reconstruction_loss_object`
  - alongside the combined `reconstruction_loss`, so both terms are visible independently during training.

### 5d. LAPO (baseline)
```bash
CUDA_VISIBLE_DEVICES=<n> MUJOCO_GL=egl python experiments/run_lapo_bc.py \
  run_id=lapo_<task>_seed1 env.name=Meta-World/masked-MT1-<task> \
  dataset.dataset_path=/tmp/slapo_local \
  logger.mode=online logger.group=lapo_reprod logger.notes="LAPO <task> seed1" \
  trainer.compile=True fabric.precision=bf16-mixed trainer.random_seed=1 \
  --stage stage_1 -cn lapo_bc_dmw_stage_1 \
  --stage stage_2 -cn lapo_bc_dmw_stage_2 \
  --stage stage_3 -cn lapo_bc_dmw_stage_3
```
- The unmasked baseline MaskLAM itself compares against - no agent or object mask anywhere in
  training. Same in-process, one-run pipeline as 5a-5c (`experiments/run_lapo_bc.py` mirrors
  `run_slapo.py`); the former subprocess-per-stage entry point is gone for this pipeline too.
- Defaults to the authors' `EpicPinkPenguin` release, pinned in `lapo_bc_dmw_stage_1/2/3.yaml`
  (`with_object_mask` stays false always - LAPO never reads it). Stage it with
  `--repo EpicPinkPenguin/...` in Section 2, same as 5a.
- MaskLAM's own paper doesn't report `sweep-into-v3` or `handle-pull-v3` (not in their MT10
  table), so for those two override `dataset.dataset_path=tsakman23/visual_masked_distracting_metaworld`
  on the command line - stage that repo instead (Section 2). Every other task uses the default above.

### 5e. IM-LAM Union (Interaction-Masked LAM)
```bash
CUDA_VISIBLE_DEVICES=<n> MUJOCO_GL=egl python experiments/run_slapo.py \
  run_id=im-lam_<task>_union_seed1 env.name=Meta-World/masked-MT1-<task> \
  dataset.dataset_path=/tmp/slapo_local \
  logger.mode=online logger.group=imlam_union logger.notes="IM-LAM union <task> seed1" \
  trainer.compile=True fabric.precision=bf16-mixed trainer.random_seed=1 \
  --stage stage_1 -cn imlam_dmw_stage_1 \
  --stage stage_2 -cn foreground_masklam_dmw_stage_2 \
  --stage stage_3 -cn foreground_masklam_dmw_stage_3
```
- IM-LAM keeps MaskLAM's IDM and swaps only the Stage-1 FDM for the directed interaction predictor
  (`IMLAMIDM` + `InteractionWorldModel`). The IDM encoder is byte-identical to MaskLAM's, so stages 2/3
  reuse `foreground_masklam_dmw_stage_2/3` (frozen encoder, same `tsakman23` data) exactly as 5b/5c.
  Stage `tsakman23/...` in Section 2. Runs under both losses, like Foreground-MaskLAM.
- **Union** (above): `imlam_dmw_stage_1` (`object_mask_loss=true`, `object_mask_input=false`) - the FDM
  loss over the agent ∪ object union, same loss axis as 5b.
- **Matched ablation (direct-z)**: swap Stage 1 to `-cn imlam_direct_z_dmw_stage_1` (feeds `z_t` straight
  to the object branch, breaking the directed embodiment->object path). Run on a **large-object** task
  (e.g. `handle-pull-v3`); on small-object tasks the object branch is near-inert and the ablation ties.
- Instrumented from step 0: per-head `beta` (`beta_msa_a/o` + `_min`), write-back norms (`proj_a/o_norm`),
  `train/total_grad_norm`, `loss_nonfinite`, and eval-time `val/object_E_O` / `val/object_prediction_ratio`
  / `val/object_R_no_transition`.
- **Pre-flight** (optional; catches compile/bf16 failures before the full run - append to the Stage-1
  command): `--selected_stages=[stage_1] ++logger.mode=offline eval=false trainer.max_epochs=null
  trainer.max_steps=60 trainer.validation_frequency=50 trainer.validation_unit=step`.

### 5f. IM-LAM Dual
```bash
CUDA_VISIBLE_DEVICES=<n> MUJOCO_GL=egl python experiments/run_slapo.py \
  run_id=im-lam_<task>_dual_seed1 env.name=Meta-World/masked-MT1-<task> \
  dataset.dataset_path=/tmp/slapo_local \
  logger.mode=online logger.group=imlam_dual logger.notes="IM-LAM dual <task> seed1" \
  trainer.compile=True fabric.precision=bf16-mixed trainer.random_seed=1 \
  --stage stage_1 -cn imlam_dual_dmw_stage_1 module.object_loss_weight=1.0 \
  --stage stage_2 -cn foreground_masklam_dmw_stage_2 \
  --stage stage_3 -cn foreground_masklam_dmw_stage_3
```
- Same interaction FDM as 5e, dual loss (`object_dual_loss=true`, `object_mask_loss=false` - mutually
  exclusive). The matched comparison to the Foreground dual baseline (5c). Sweep `module.object_loss_weight`
  (`lambda_O`) from the CLI (default 1.0); per-term `reconstruction_loss_agent`/`_object` log separately.
  The direct-z ablation and pre-flight apply the same as 5e (swap `-cn imlam_direct_z_dmw_stage_1`).

#### Stage-1 mechanism diagnostics (run after a union/dual/ablation run finishes)
Compute the object-prediction / object-dynamics-probe / agent-path metrics on the frozen Stage-1
checkpoint and push them into that run's W&B summary (under `eval/`), so IM-LAM / Foreground / direct-z
compare directly in the runs table:
```bash
CUDA_VISIBLE_DEVICES=<n> python scripts/imlam_diagnostics/run_diagnostics.py \
  --checkpoint checkpoints/im-lam_<task>_union_seed1-1/<latest-step>.ckpt \
  --model imlam --loss union --task <task> \
  --data-path /tmp/slapo_local \
  --wandb-run-id im-lam_<task>_union_seed1
```
- Run once per checkpoint: IM-LAM union/dual, the same-loss Foreground baseline (`--model foreground`),
  and the direct-z ablation (`--config-name imlam_direct_z_dmw_stage_1`). `--max-batches` bounds cost/RAM
  (default 64; the full test split is ~100k samples). Diagnostics 1+3 use a shuffled loader, the probe a
  sequential one.

---

### Visualize MaskLAM vs Foreground-MaskLAM reconstruction

After both Stage-1 checkpoints exist, compare their FDM reconstructions on identical push frames:
```bash
CUDA_VISIBLE_DEVICES=<n> MUJOCO_GL=egl python scripts/compare_reconstruction.py \
  --env Meta-World/masked-MT1-<task> \
  --masklam-ckpt checkpoints/slapo_q1_Meta-World_masked-MT1-<task>_seed1-1 \
  --fg-ckpt      checkpoints/fg_masklam_<task>_seed1-1 \
  --num-frames 6
```
Writes per-frame panels `[GT | MaskLAM pred | FG pred | union overlay | MaskLAM-extra-error]` plus
`fg_vs_masklam_mse.txt` under `docs/figures/fg_vs_masklam/`. Expected: object-region MSE much higher
for MaskLAM than FG; agent-region MSE comparably low for both. Reads frames from the object-mask
dataset, so stage `<task>` (Section 2) first.

---


#### Misc / utilities

```bash
# Reclaim disk: delete stale partial HF downloads (ONLY when nothing is downloading)
find /data2/masklam/datasets/hf_home/hub -name '*.incomplete' -delete

# Free a staged copy from RAM when done with a task
rm -rf /tmp/slapo_local/<task>
```
