# EXPO — Expressive Policy Optimization

EXPO fine-tunes a frozen diffusion policy via online RL using a small Gaussian edit policy and Q-ensemble, applied here to the **Robomimic Square** (low-dim) task.

## Installation

```bash
# 1. System dependencies (MuJoCo rendering)
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# 2. Create conda environment
conda env create -f conda_environment.yaml
conda activate robodiff

# 3. Install diffusion_policy package (editable)
pip install -e .
```

## Dataset

Download the Robomimic square dataset and place it under `data/robomimic/datasets/`:

```bash
# Create directory
mkdir -p data/robomimic/datasets/square/ph

# Download proficient-human (ph) dataset
wget -O data/robomimic/datasets/square/ph/low_dim.hdf5 \
  https://robomimic.github.io/datasets/square/ph/low_dim.hdf5
```

> **Note:** The dataset is not included in this repo. See the [robomimic datasets page](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html) for other tasks (lift, can, transport, tool_hang).

The pre-trained base diffusion policy checkpoint is already included at `data/experiments/`.

## Training

```bash
python train_expo_square.py \
    --base_checkpoint data/experiments/low_dim/square_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt \
    --output_dir data/expo_square_output \
    --device cuda:0
```

Key arguments:

| Flag | Default | Description |
|---|---|---|
| `--beta` | `0.05` | Edit scale bound |
| `--N` | `8` | Action candidates per step |
| `--utd_ratio` | `20` | Critic updates per env step |
| `--max_steps` | `100000` | Total online environment steps |
| `--eval_interval` | `3000` | Steps between evaluations |
| `--seed` | `42` | Random seed |

Run `python train_expo_square.py --help` for all options.

## Evaluation

```bash
python eval_expo.py \
    --checkpoint data/expo_square_output/checkpoints/latest.ckpt \
    --output_dir data/expo_eval_output \
    --device cuda:0
```

Outputs `eval_log.json` (test scores) and `media/` (rollout videos).
