# Distracting MetaWorld Dataset

This repository generates a distracting-vision version of the Meta-World benchmark and uploads it as a standard Hugging Face dataset.  
Each task is exported as a separate dataset configuration with clean RGB observations, DAVIS-background–distracted observations, and robot-arm segmentation masks.

## Environment setup

### 1. Create and activate a conda environment

```bash
conda create -n distracting-metaworld python=3.10
conda activate distracting-metaworld
```

Any recent Python 3.9–3.11 version should work; `3.10` is what we test with.

### 2. Install Python requirements

From the repository root:

```bash
pip install -r requirements.txt
```

This installs Meta-World, Gymnasium, Hugging Face `datasets`, and a few utility libraries used by the generation script.

### 3. (Optional but recommended) Hugging Face authentication

If you want to upload directly to the Hugging Face Hub (using `--push_to_hub`):

```bash
pip install "huggingface_hub"
hf auth login   # or set HF_TOKEN in your environment
```

The default repository ID is `EpicPinkPenguin/visual_distracting_metaworld` (see `REPO_ID` in `generate_dataset_huggingface.py`); you can change this in the script if needed.

### 4. (Optional) Background dataset (DAVIS)

By default the script expects a DAVIS video dataset at `datasets/DAVIS`.  
If yours lives elsewhere, pass `--background_dataset_path /path/to/DAVIS` to the generation script.

## Generating datasets

All dataset generation is driven by `generate_dataset_huggingface.py`.  
The key arguments are:

- `--env`: Meta-World environment ID, e.g. `Meta-World/MT1-bin-picking-v3`
- `--split`: `train` or `test`
- `--num_steps`: maximum total timesteps to generate (overrides episode count)
- `--seed`: base random seed (different seeds for train/test to avoid overlap)
- `--push_to_hub`: actually upload the dataset to the Hub (otherwise stays local)

### Single task: train and test splits

To generate a **train** split for a single task (here: MT1 bin-picking) with 1M steps:

```bash
python generate_dataset_huggingface.py \
  --env Meta-World/MT1-bin-picking-v3 \
  --split train \
  --num_steps 1000000 \
  --seed 0 \
  --push_to_hub
```

To generate the corresponding **test** split with 100k steps and a different seed:

```bash
python generate_dataset_huggingface.py \
  --env Meta-World/MT1-bin-picking-v3  \
  --split test \
  --num_steps 100000 \
  --seed 100000 \
  --push_to_hub
```

If you want to test your setup without uploading or generating many frames, add `--dry_run` (it generates ~20 frames and exits):

```bash
python generate_dataset_huggingface.py \
  --env Meta-World/MT1-bin-picking-v3 \
  --split train \
  --num_steps 1000 \
  --seed 0 \
  --dry_run
```

### Full MT50 dataset (all tasks)

To generate both train and test splits for **all 50 Meta-World MT50 tasks**, use the helper script:

```bash
bash generate_all_datasets.sh
```

This script:

- Loops over an explicit list of 50 MT50 v3 tasks
- For each task, calls `generate_dataset_huggingface.py` twice:
  - Train split: `--split train --num_steps 1000000 --seed 0 --push_to_hub`
  - Test split:  `--split test  --num_steps 100000 --seed 100000 --push_to_hub`

You can edit `NUM_STEPS_TRAIN`, `NUM_STEPS_TEST`, `TRAIN_SEED`, and `TEST_SEED` at the top of `generate_all_datasets.sh` if you want different lengths or seeds.

> **Note:** Generating the full dataset for all tasks with 1M train and 100k test steps each is computationally expensive and will take a long time. Make sure you have sufficient compute, storage, and a valid Hugging Face token before running the full script.

