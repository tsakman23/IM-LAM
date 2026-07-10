# Experiment commands

Reference for the commands used in this project, grouped by purpose. Replace
`<task>` with a task slug, e.g. `push-v3`, `push-wall-v3`, `plate-slide-v3`,
`door-open-v3`, `faucet-open-v3`, `handle-pull-v3`, `dial-turn-v3`,
`pick-place-v3`, `peg-insert-side-v3`, `sweep-into-v3`.

## 0. Environment (prerequisites for everything below)

- Interpreter: Activate the conda env with `conda activate ./conda_env`.
- Always export:
  ```bash
  export MUJOCO_GL=egl                              # headless EGL rendering
  export HF_HOME=<YOUR CWD>/datasets/hf_home    # keep HF caches off home dir
  ```
- HuggingFace auth: add **write** token in `~/.bashrc` (`HF_TOKEN`). Confirm with
  `HF_HOME=<YOUR CWD>/datasets/hf_home hf auth whoami` -> should print your username.
- Two repos are involved:
  - **`EpicPinkPenguin/visual_distracting_metaworld`** - the MaskLAM authors' release (agent mask only).
  - **`tsakman23/visual_masked_distracting_metaworld`** - my regenerated data (adds `object_mask` + `object_state`).

---

## 1. Download a task dataset from MaskLAM's HF repo (authors' release)

Pre-fetch + build into the local cache so training just cache-hits (standalone, also
avoids the train-time build deadlock). Safe to Ctrl+C and rerun (resumes).

```bash
python scripts/download_dataset.py \
  --env Meta-World/masked-MT1-<task>
```
Both splits are fetched by default. `--cache-dir` defaults to `./datasets/slapo/<env>` and
**must match** the run's `dataset.cache_dir`.

---

## 2. Generate + push an object-mask dataset to my repo

Object masks are ground-truth simulator masks emitted *by the generator itself* (columns
`object_mask` + `object_state`) - there is no separate "generate masks" step, and SAM is not
currently supported as an alternative to ground-truth (GT) simulator masks. Run from the generator repo. `--push_to_hub` uploads to
`tsakman23/visual_masked_distracting_metaworld` (config = task slug).

```bash
cd distracting-metaworld-dataset-main
# train: 1M steps, seed 0
python generate_dataset_huggingface.py \
  --env Meta-World/MT1-<task> --split train --num_steps 1000000 --seed 0 --push_to_hub
# test: 100k steps, seed 100000  (can run in parallel)
python generate_dataset_huggingface.py \
  --env Meta-World/MT1-<task> --split test --num_steps 100000 --seed 100000 --push_to_hub
```
- ~5 h per 1M-step split. Drop `--push_to_hub` to stay local; the cache is kept at
  `datasets/hf_cache/<task>_<split>`.
- Articulated objects (door/faucet/handle/dial): if the object mask looks wrong, add the task to
  `TASK_OBJECT_OVERRIDES` in `generate_dataset_huggingface.py`; `push`/`door-open` work with the
  default heuristic.

### 2a. Re-push from cache if a `--push_to_hub` upload failed (e.g. auth) - no regeneration
```bash
cd distracting-metaworld-dataset-main
python verify/repush_from_cache.py --config <task> --split train
# ...and --split test
```

---

## 3. Verify a generated / downloaded dataset

```bash
cd distracting-metaworld-dataset-main
# IID vs the authors' release (8 shared columns)
# generates a fresh sample or streams --ours-repo
# Checks first 3k rows of the split (train/test) for equality. If you want to check the full dataset, increase `--n`.
python verify/iid_check.py --config <task> --split train --n 3000

# Visual mask panels (observation | agent mask | object mask | overlay) for eyeballing
python verify/render_mask_panels.py \
  --env Meta-World/MT1-<task> --steps 150 --num_frames 8 --out datasets/verify/mask_panels/<task>
```

---

## 4. Run MaskLAM (baseline) on the authors' dataset - full 3-stage pipeline

```bash
CUDA_VISIBLE_DEVICES=<n> MUJOCO_GL=egl python experiments/run_slapo.py \
  run_id=masklam_<task>_seed1 \
  env.name=Meta-World/masked-MT1-<task> \
  logger.mode=online \
  logger.group=masklam_reprod \
  logger.notes="MaskLAM <task>" \
  trainer.compile=True \
  fabric.precision=bf16-mixed \
  trainer.random_seed=1 \
  --stage stage_1 -cn slapo_dmw_stage_1 \
  --stage stage_2 -cn slapo_dmw_stage_2 \
  --stage stage_3 -cn slapo_dmw_stage_3
```

- **Dataset:** pinned in-config, not on the CLI - `slapo_dmw_stage_1/2/3.yaml` set
  `dataset.dataset_path: EpicPinkPenguin/visual_distracting_metaworld` (the authors' agent-mask-only
  release) and `dataset.with_object_mask: false`. Nothing dataset-related is passed on the command line.
- **Cache:** `dataset.cache_dir` now defaults to `./datasets/slapo/${env.name}` (matching the
  Section-1 pre-download path), so no override is needed - pre-fetch with the Section-1 command
  first and Stage 1 cache-hits instead of downloading. `logs/run_slapo_dmw.sh` wraps this command
  as a launcher - TODO: clean that up later.

---

## 5. Run Foreground-MaskLAM (baseline)

Same pipeline, but Stage 1 uses `foreground_masklam_dmw_stage_1` (object-mask union loss, data
from my repo). Stages 2/3 are the standard MaskLAM configs.
```bash
# recommended: pre-fetch the object-mask data first (avoids the train-time build deadlock)
python scripts/download_dataset.py \
  --env Meta-World/masked-MT1-<task> \
  --repo tsakman23/visual_masked_distracting_metaworld --with-object-mask

CUDA_VISIBLE_DEVICES=<n> MUJOCO_GL=egl python experiments/run_slapo.py \
  run_id=fg_masklam_<task>_seed1 \
  env.name=Meta-World/masked-MT1-<task> \
  logger.mode=online \
  logger.group=foreground_masklam \
  logger.notes="Foreground-MaskLAM <task> seed1" \
  trainer.compile=True \
  fabric.precision=bf16-mixed \
  trainer.random_seed=1 \
  --stage stage_1 -cn foreground_masklam_dmw_stage_1 \
  --stage stage_2 -cn slapo_dmw_stage_2 \
  --stage stage_3 -cn slapo_dmw_stage_3
```

- **Dataset:** `foreground_masklam_dmw_stage_1` inherits `slapo_dmw_stage_1` and overrides
  `dataset.dataset_path: tsakman23/visual_masked_distracting_metaworld` + `dataset.with_object_mask: true`
  in-config - that swaps in my object-mask repo for Stage 1. Stages 2/3 reuse the stock
  `slapo_dmw_stage_2/3` unchanged, so they train on the authors' agent-mask release (same
  trajectories, minus the unused object-mask column).
- **Cache:** `dataset.cache_dir` defaults to `./datasets/slapo/${env.name}` (set in `sl_default`,
  matching `download_dataset.py`'s default), so a pre-fetched split cache-hits with no per-run CLI
  override. HuggingFace namespaces the cache by repo id, so `EpicPinkPenguin___...` and
  `tsakman23___...` land in separate subdirs and never collide.
- Regenerated tasks with object masks currently on `tsakman23/visual_masked_distracting_metaworld`:
  `push-v3`, `door-open-v3`, `sweep-into-v3` have **both** train + test (runnable end-to-end);
  `handle-pull-v3` and `pick-place-v3` are **test-only** (Stage 1 needs train, so not yet runnable).
  For a task other than push, add `env.name=Meta-World/masked-MT1-<task>` to the run.
- The config runs Foreground-MaskLAM as **loss-only**: `object_mask_loss=true`, `object_mask_input=false`.
  The agent U object union gates the FDM reconstruction loss (so the decoder must reconstruct the
  object), while the encoder/IDM mask input stays agent-only - so `z_t` remains an embodiment action
  and the frozen encoder stays consistent with stages 2/3 (which feed it the agent mask).
- `object_mask_input=true` is an opt-in experiment only (feed the union into the encoder too). It
  changes what `z_t` encodes and would require threading the union through stages 2/3 - don't enable
  it without that.

---

## 6. Misc / utilities

```bash
# Reclaim disk: delete stale partial downloads (ONLY when nothing is downloading)
find /data2/masklam/datasets/hf_home/hub -name '*.incomplete' -delete
```
