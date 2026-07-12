# Local dataset creation + loading (skip the HF round-trip) - Design

Date: 2026-07-10
Status: Approved for implementation

## Goal
Let the Meta-World dataset generator persist a **locally loadable** copy of a generated
dataset so training can read it directly, without the generate -> push-to-HF -> download-back
round-trip that is pure overhead when generation and training happen on the same machine.
The HF path stays fully intact (portable/reproducible distribution); local loading is an
explicit opt-in.

## Background (verified)
- Generator (`distracting-metaworld-dataset-main/generate_dataset_huggingface.py`) builds a
  dataset via `Dataset.from_generator(cache_dir=datasets/hf_cache/<task>_<split>)`. That cache
  is a `from_generator` scratch cache (not a loadable dataset dir) and, on `--push_to_hub`, it
  is **deleted** after a successful upload.
- Training loads via `ifo/common/data/huggingface.py::HuggingFaceDataset.__init__`, which calls
  `load_dataset(path=<dataset_path>, name=<config>, split=<split>, ...)`. `dataset_path` is the
  HF repo id (e.g. `tsakman23/visual_masked_distracting_metaworld`); `name` is the bare config
  (e.g. `push-v3`).
- `MetaWorldDataset` post-processes after load: renames `observation_distracted`->`observation`
  for distracted/masked variants, handles SAM masks, selects `columns`, and applies
  `with_format("torch", columns=...)`. This logic must run identically for local loads.
- The generator is tracked inside the main repo (not a submodule), so both edits live in one repo.

## Layout (contract between generator and training)
```
<root>/<config>/<split>/        e.g. datasets/slapo_local/push-v3/train/
```
One `save_to_disk` directory per (config, split); one root can hold many tasks/splits. All
columns are preserved exactly (no PNG re-encode/decode).

## Generator side
- Add `--save_local [DIR]` (default root `datasets/slapo_local`), **additive** to
  `--push_to_hub` (either or both may be passed).
- After building `dataset`, if `--save_local` is set: `dataset.save_to_disk(<root>/<task>/<split>)`.
- Scratch-cache cleanup rule: delete the `from_generator` cache once the data is safely
  persisted by **either** a push **or** a local save (so `--save_local` alone leaves no
  redundant scratch copy). If neither flag is set, behavior is unchanged (cache kept + message).

## Training side
- In `HuggingFaceDataset.__init__`: if `path` is an existing local directory, load
  `os.path.join(path, name, split)` via `datasets.load_from_disk` instead of
  `load_dataset(repo_id, name, split)`. Everything downstream is unchanged.
- **No auto-detect / no magic.** Opt in explicitly by setting
  `dataset.dataset_path=./datasets/slapo_local` (a local dir). A repo id loads from HF exactly
  as today - zero behavior change for existing configs.

## Usage
```bash
# generate locally, no HF round-trip:
python generate_dataset_huggingface.py --env Meta-World/MT1-<task> --split train --save_local
# train reading it directly:
python experiments/run_slapo.py ... dataset.dataset_path=./datasets/slapo_local \
  --stage stage_1 -cn foreground_masklam_dmw_stage_1
```

## Non-goals
- No converter for already-generated caches or the already-downloaded push-v3 (future
  generations only; a converter can be added later if the existing ~110 GB of caches are worth
  salvaging).
- No auto-preference of local over HF; the user chooses via `dataset_path`.
- No change to the HF push path or to Stages 2/3.

## Files touched
- `distracting-metaworld-dataset-main/generate_dataset_huggingface.py` - `--save_local` flag,
  `save_to_disk`, cache-cleanup rule.
- `ifo/common/data/huggingface.py` - local `load_from_disk` branch in `HuggingFaceDataset`.

## Testing
- Training branch (unit): build a tiny synthetic dataset, `save_to_disk` to
  `<root>/<config>/<split>`; assert `HuggingFaceDataset` loads it via the local branch, that
  `columns`/`with_format` still apply, and that a non-local `path` is unaffected (repo-id path
  untouched). Covers the `MetaWorldDataset` rename by including an `observation_distracted`
  column in the synthetic data.
- Generator (light): a `save_to_disk` layout check on a few episodes writes the expected
  `<root>/<task>/<split>` directory that `load_from_disk` can reopen.

## Verification
- Round-trip: generate a few episodes with `--save_local` to a temp root, then load through
  `get_dataset(dataset_path=<root>, with_object_mask=true)` and confirm the sample has the
  expected keys/shapes (observation, mask, object_mask, action) - no network access.
```
