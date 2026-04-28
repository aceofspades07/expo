"""
Evaluate a trained EXPO edit policy on Robomimic Square (lowdim).

Mirrors the interface of eval.py exactly:
    python eval_expo.py \
        --checkpoint data/expo_square_output/checkpoints/latest.ckpt \
        --output_dir data/expo_eval_output \
        --device cuda:0

The EXPO checkpoint contains:
  - Path to the base diffusion policy checkpoint
  - Edit policy, Q-ensemble, target critic, alpha state dicts
  - Full config (obs_dim, action_dim, n_obs_steps, n_action_steps, beta, N, etc.)

Output is identical to eval.py: eval_log.json + media/ videos.
"""

import sys
import os
import pathlib
import json

# Line-buffered I/O
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import click
import hydra
import torch
import torch.nn as nn
import numpy as np
import dill
import wandb

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.expo.expo_learner import EXPOLearner


# ═══════════════════════════════════════════════════════════════════════════
#  EXPOPolicyWrapper — BaseLowdimPolicy-compatible wrapper for env runner
# ═══════════════════════════════════════════════════════════════════════════
class EXPOPolicyWrapper(nn.Module):
    """
    Wraps EXPOLearner to match the BaseLowdimPolicy interface expected by
    RobomimicLowdimRunner.run(policy).

    predict_action(obs_dict) → {"action": (B, Ta, Da)}
    """

    def __init__(self, learner: EXPOLearner):
        super().__init__()
        self.learner = learner
        self._dummy = nn.Parameter(torch.empty(0, device=learner.device))

    @property
    def device(self):
        return self.learner.device

    @property
    def dtype(self):
        return torch.float32

    def reset(self):
        pass

    def predict_action(self, obs_dict: dict) -> dict:
        obs_multi = obs_dict["obs"]  # (B, To, Do)
        B = obs_multi.shape[0]
        obs_flat = obs_multi[:, -1, :]  # (B, Do)

        with torch.no_grad():
            a_star_flat = self.learner._sample_otf_action(
                obs_flat, obs_multi, use_target=True
            )

        action = a_star_flat.reshape(
            B, self.learner.n_action_steps, self.learner.action_dim
        )
        return {"action": action}


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
@click.command()
@click.option("-c", "--checkpoint", required=True, help="Path to EXPO .ckpt file")
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(
            f"Output path {output_dir} already exists! Overwrite?", abort=True
        )
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)

    # ── Load EXPO checkpoint ───────────────────────────────────────
    print("Loading EXPO checkpoint …")
    expo_data = torch.load(open(checkpoint, "rb"), map_location=dev)
    expo_cfg = expo_data["cfg"]
    base_checkpoint_path = expo_data["base_checkpoint_path"]

    print(f"  Base checkpoint: {base_checkpoint_path}")
    print(f"  EXPO config: β={expo_cfg['beta']}, N={expo_cfg['N']}, "
          f"obs_dim={expo_cfg['obs_dim']}, action_dim={expo_cfg['action_dim']}")

    # ── Load frozen base policy ────────────────────────────────────
    print("Loading frozen base diffusion policy …")
    base_payload = torch.load(
        open(base_checkpoint_path, "rb"), pickle_module=dill
    )
    base_cfg = base_payload["cfg"]

    cls = hydra.utils.get_class(base_cfg._target_)
    workspace = cls(base_cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(base_payload, exclude_keys=None, include_keys=None)

    base_policy = (
        workspace.ema_model if base_cfg.training.use_ema else workspace.model
    )
    base_policy.to(dev)
    base_policy.eval()
    for p in base_policy.parameters():
        p.requires_grad_(False)

    # ── Build EXPO learner and load trained weights ────────────────
    learner = EXPOLearner(
        base_policy=base_policy,
        obs_dim=int(expo_cfg["obs_dim"]),
        action_dim=int(expo_cfg["action_dim"]),
        n_obs_steps=int(expo_cfg["n_obs_steps"]),
        n_action_steps=int(expo_cfg["n_action_steps"]),
        device=dev,
        beta=float(expo_cfg["beta"]),
        N=int(expo_cfg["N"]),
    )
    learner.load(checkpoint)
    learner.edit_policy.eval()
    print("  EXPO learner loaded successfully.")

    # ── Build env runner (same as eval.py) ─────────────────────────
    print("Building environment runner …")
    env_runner = hydra.utils.instantiate(
        base_cfg.task.env_runner, output_dir=output_dir
    )

    # ── Run evaluation ─────────────────────────────────────────────
    print("Running evaluation …")
    wrapped_policy = EXPOPolicyWrapper(learner)
    runner_log = env_runner.run(wrapped_policy)

    # ── Dump log to JSON (identical format to eval.py) ─────────────
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)

    # ── Print summary ──────────────────────────────────────────────
    mean_score = json_log.get("test/mean_score", "N/A")
    print(f"\n{'='*60}")
    print(f"  Evaluation complete!")
    print(f"  test/mean_score = {mean_score}")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
