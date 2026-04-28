"""
Train EXPO edit policy on Robomimic Square (lowdim) with a frozen
diffusion-policy base.

Usage:
    python train_expo_square.py \
        --base_checkpoint data/experiments/low_dim/square_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt \
        --output_dir data/expo_square_output \
        --device cuda:0 \
        --beta 0.05 \
        --max_steps 100000

The base diffusion policy is loaded from --base_checkpoint and kept
completely frozen.  Only the edit policy, Q-ensemble, and entropy
temperature are trained.

ALL data storage and sampling happens on CUDA — the only numpy↔tensor
boundary is the env step (MuJoCo gives numpy, we convert once).
"""

import sys
import os
import pathlib
import json
import copy
import math
import collections

# Line-buffered I/O
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import click
import hydra
import torch
import torch.nn as nn
import numpy as np
import dill
import tqdm
import h5py

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.expo.expo_learner import EXPOLearner
from diffusion_policy.expo.replay_buffer import ReplayBuffer


# ═══════════════════════════════════════════════════════════════════════════
#  Load frozen base policy
# ═══════════════════════════════════════════════════════════════════════════
def load_frozen_base(checkpoint_path: str, device: str):
    """Load pre-trained diffusion policy, freeze it, return (policy, cfg)."""
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir="/tmp/_expo_base_tmp")
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    return policy, cfg


# ═══════════════════════════════════════════════════════════════════════════
#  Build single Robomimic Square env (non-vectorised)
# ═══════════════════════════════════════════════════════════════════════════
def make_env(cfg):
    """Create a single Robomimic Square lowdim environment."""
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import (
        RobomimicLowdimWrapper,
    )
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    from diffusion_policy.model.common.rotation_transformer import (
        RotationTransformer,
    )

    task_cfg = cfg.task
    dataset_path = os.path.expanduser(task_cfg.env_runner.dataset_path)
    obs_keys = list(task_cfg.env_runner.obs_keys)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    rotation_transformer = None
    if task_cfg.env_runner.get("abs_action", False):
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )

    n_obs_steps = int(task_cfg.env_runner.n_obs_steps)
    n_latency_steps = int(task_cfg.env_runner.get("n_latency_steps", 0))
    n_action_steps = int(task_cfg.env_runner.n_action_steps)
    max_steps = int(task_cfg.env_runner.max_steps)

    env = MultiStepWrapper(
        RobomimicLowdimWrapper(
            env=robomimic_env,
            obs_keys=obs_keys,
            init_state=None,
            render_hw=(128, 128),
        ),
        n_obs_steps=n_obs_steps + n_latency_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
    )
    return env, rotation_transformer


# ═══════════════════════════════════════════════════════════════════════════
#  Offline dataset — CUDA-resident
# ═══════════════════════════════════════════════════════════════════════════
class OfflineDataset:
    """
    Robomimic HDF5 dataset stored entirely as CUDA tensors.
    `.sample()` returns a dict of CUDA tensors — zero CPU involvement.
    """

    def __init__(
        self,
        dataset_path: str,
        obs_keys: list,
        abs_action: bool,
        n_action_steps: int,
        device: torch.device,
    ):
        from diffusion_policy.model.common.rotation_transformer import (
            RotationTransformer,
        )

        rotation_transformer = None
        if abs_action:
            rotation_transformer = RotationTransformer(
                "axis_angle", "rotation_6d"
            )

        with h5py.File(dataset_path, "r") as f:
            demos = sorted(f["data"].keys())
            all_obs = []
            all_actions = []
            all_rewards = []
            all_next_obs = []
            all_dones = []
            all_masks = []

            for demo_key in demos:
                demo = f[f"data/{demo_key}"]
                obs_parts = [demo["obs"][k][:] for k in obs_keys]
                obs = np.concatenate(obs_parts, axis=-1).astype(np.float32)

                actions = demo["actions"][:].astype(np.float32)

                # Transform actions: axis_angle (7d) → rotation_6d (10d)
                if abs_action and rotation_transformer is not None:
                    pos = actions[:, :3]
                    rot_aa = actions[:, 3:6]
                    gripper = actions[:, 6:7]
                    rot_6d = rotation_transformer.forward(
                        torch.from_numpy(rot_aa)
                    ).numpy()
                    actions = np.concatenate(
                        [pos, rot_6d, gripper], axis=-1
                    ).astype(np.float32)

                rewards = demo["rewards"][:].astype(np.float32)
                dones = demo["dones"][:].astype(np.float32)

                T = len(obs)
                for t in range(T - 1):
                    all_obs.append(obs[t])
                    all_actions.append(actions[t])
                    all_rewards.append(rewards[t])
                    all_next_obs.append(obs[t + 1])
                    done = dones[t]
                    all_dones.append(done)
                    all_masks.append(1.0 - done)

        # Build numpy then move to CUDA once
        raw_actions = np.array(all_actions, dtype=np.float32)
        tiled_actions = np.tile(raw_actions, (1, n_action_steps))

        self.observations      = torch.from_numpy(np.array(all_obs, dtype=np.float32)).to(device)
        self.actions           = torch.from_numpy(tiled_actions).to(device)
        self.rewards           = torch.from_numpy(np.array(all_rewards, dtype=np.float32)).to(device)
        self.next_observations = torch.from_numpy(np.array(all_next_obs, dtype=np.float32)).to(device)
        self.dones             = torch.from_numpy(np.array(all_dones, dtype=np.float32)).to(device)
        self.masks             = torch.from_numpy(np.array(all_masks, dtype=np.float32)).to(device)
        self.size = self.observations.shape[0]
        self.device = device

        print(
            f"  Offline dataset loaded to {device}: {self.size} transitions, "
            f"obs_dim={self.observations.shape[1]}, "
            f"raw_action_dim={raw_actions.shape[1]}, "
            f"flat_action_dim={self.actions.shape[1]} "
            f"(tiled x{n_action_steps})"
        )

    def sample(self, batch_size: int) -> dict:
        """Return a dict of CUDA tensors."""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "observations":      self.observations[idx],
            "actions":           self.actions[idx],
            "rewards":           self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "dones":             self.dones[idx],
            "masks":             self.masks[idx],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  EXPOPolicyWrapper — makes EXPO learner compatible with env_runner.run()
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
            # Sample ONE base action (no critic selection — base is 0.98)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                result = self.learner.base_policy.predict_action(obs_dict)
            base_a = result["action"].float().reshape(B, -1)  # (B, flat_action_dim)

            # Apply the learned edit (deterministic = use mean, not sample)
            delta_dist = self.learner.edit_policy(obs_flat, base_a)
            # Use tanh(mean) for deterministic eval — rsample() adds noise
            # that corrupts the near-optimal base policy
            delta_mean = torch.tanh(delta_dist.base_dist.mean)
            a_star = base_a + self.learner.beta * delta_mean

        action = a_star.reshape(
            B, self.learner.n_action_steps, self.learner.action_dim
        )
        return {"action": action}


# ═══════════════════════════════════════════════════════════════════════════
#  Undo rotation transform for absolute action spaces
# ═══════════════════════════════════════════════════════════════════════════
def undo_transform_action(action, rotation_transformer):
    """Convert rotation_6d back to axis_angle for env stepping."""
    if rotation_transformer is None:
        return action
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape(-1, 2, 10)
    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3 : 3 + d_rot]
    gripper = action[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)
    if raw_shape[-1] == 20:
        uaction = uaction.reshape(*raw_shape[:-1], 14)
    return uaction


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation using RobomimicLowdimRunner (same as eval.py)
# ═══════════════════════════════════════════════════════════════════════════
def run_eval(cfg, learner, output_dir, device):
    """Run full evaluation using the existing RobomimicLowdimRunner."""
    import wandb

    eval_output = os.path.join(output_dir, "eval_tmp")
    os.makedirs(eval_output, exist_ok=True)

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner, output_dir=eval_output
    )

    wrapped = EXPOPolicyWrapper(learner)
    runner_log = env_runner.run(wrapped)

    mean_score = runner_log.get("test/mean_score", 0.0)
    return mean_score, runner_log


# ═══════════════════════════════════════════════════════════════════════════
#  Main training loop — everything on CUDA
# ═══════════════════════════════════════════════════════════════════════════
@click.command()
@click.option("-c", "--base_checkpoint", required=True, help="Path to base diffusion policy .ckpt")
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--beta", default=0.05, type=float, help="Edit scale bound β")
@click.option("--N", "n_candidates", default=8, type=int, help="Action candidates N")
@click.option("--utd_ratio", default=20, type=int, help="Critic updates per env step")
@click.option("--max_steps", default=100_000, type=int, help="Total online env steps")
@click.option("--eval_interval", default=3_000, type=int, help="Steps between evaluations")
@click.option("--batch_size", default=256, type=int)
@click.option("--offline_ratio", default=0.5, type=float, help="Fraction of offline data in each batch")
@click.option("--start_training", default=1000, type=int, help="Random warm-up steps before training")
@click.option("--lr", default=3e-4, type=float)
@click.option("--seed", default=42, type=int)
@click.option("--inference_steps", default=4, type=int, help="DDIM inference steps for base policy (default 4, was 100 DDPM)")
def main(
    base_checkpoint,
    output_dir,
    device,
    beta,
    n_candidates,
    utd_ratio,
    max_steps,
    eval_interval,
    batch_size,
    offline_ratio,
    start_training,
    lr,
    seed,
    inference_steps,
):
    # ── Setup ──────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    dev = torch.device(device)

    # ── Load frozen base policy ────────────────────────────────────
    print("=" * 60)
    print("Loading frozen base diffusion policy …")
    base_policy, cfg = load_frozen_base(base_checkpoint, device)
    obs_dim         = int(cfg.obs_dim)
    action_dim      = int(cfg.action_dim)
    n_obs_steps     = int(cfg.n_obs_steps)
    n_action_steps  = int(cfg.n_action_steps)
    n_latency_steps = int(cfg.n_latency_steps)
    abs_action      = bool(cfg.task.get("abs_action", False))
    flat_action_dim = action_dim * n_action_steps

    # ── PERF: Swap DDPM→DDIM scheduler (100→8 denoising steps) ────
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    old_sched = base_policy.noise_scheduler
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=old_sched.config.num_train_timesteps,
        beta_start=old_sched.config.beta_start,
        beta_end=old_sched.config.beta_end,
        beta_schedule=old_sched.config.beta_schedule,
        clip_sample=old_sched.config.clip_sample,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type=old_sched.config.prediction_type,
    )
    base_policy.noise_scheduler = ddim_scheduler
    base_policy.num_inference_steps = inference_steps
    print(f"  Swapped DDPM→DDIM scheduler, inference_steps={inference_steps}")

    # NOTE: torch.compile skipped — incompatible with einops.rearrange
    # on this PyTorch version (SymInt hashing issue)

    # ── PERF: Convert frozen base policy to float16 ──────────────
    base_policy.half()
    # Keep normalizer in float32 for precision
    if hasattr(base_policy, 'normalizer'):
        base_policy.normalizer.float()
    print("  Base policy converted to float16")

    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, "
          f"n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")
    print(f"  flat_action_dim={flat_action_dim}, abs_action={abs_action}")
    print(f"  β={beta}, N={n_candidates}, UTD={utd_ratio}")
    print("=" * 60)

    # ── Build learner ──────────────────────────────────────────────
    learner = EXPOLearner(
        base_policy=base_policy,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        device=dev,
        beta=beta,
        N=n_candidates,
        utd_ratio=utd_ratio,
        lr=lr,
        batch_size=batch_size,
    )

    # ── Build env ──────────────────────────────────────────────────
    print("Building Robomimic Square environment …")
    env, rotation_transformer = make_env(cfg)

    # ── Load offline dataset (directly to CUDA) ────────────────────
    print("Loading offline dataset for batch blending …")
    dataset_path = os.path.expanduser(cfg.task.env_runner.dataset_path)
    obs_keys     = list(cfg.task.env_runner.obs_keys)
    offline_dataset = OfflineDataset(
        dataset_path, obs_keys, abs_action=abs_action,
        n_action_steps=n_action_steps,
        device=dev,
    )

    # ── Online replay buffer (CUDA-resident) ───────────────────────
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        action_dim=flat_action_dim,
        capacity=max_steps + 10_000,
        device=dev,
    )

    # ── Pre-allocate combined batch tensors on CUDA ────────────────
    offline_bs = int(batch_size * offline_ratio)
    online_bs  = batch_size
    total_bs   = online_bs + offline_bs

    combined = {
        "observations":      torch.empty(total_bs, obs_dim, device=dev),
        "actions":           torch.empty(total_bs, flat_action_dim, device=dev),
        "rewards":           torch.empty(total_bs, device=dev),
        "next_observations": torch.empty(total_bs, obs_dim, device=dev),
        "dones":             torch.empty(total_bs, device=dev),
        "masks":             torch.empty(total_bs, device=dev),
    }
    combined_next_multi = torch.empty(
        total_bs, n_obs_steps, obs_dim, device=dev
    )

    # ── Training loop ──────────────────────────────────────────────
    print(f"\nStarting online training for {max_steps} steps …\n")

    obs = env.reset()  # (To, Do) numpy from MultiStepWrapper
    episode_return = 0.0
    episode_count  = 0
    episode_step   = 0
    all_metrics    = []

    expo_cfg = {
        "obs_dim":         obs_dim,
        "action_dim":      action_dim,
        "n_obs_steps":     n_obs_steps,
        "n_action_steps":  n_action_steps,
        "beta":            beta,
        "N":               n_candidates,
        "utd_ratio":       utd_ratio,
        "lr":              lr,
        "batch_size":      batch_size,
        "seed":            seed,
        "base_checkpoint": base_checkpoint,
    }

    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=1)
    env_future = None

    pbar = tqdm.tqdm(range(max_steps), desc="EXPO Training")
    for step in pbar:

        # ── Collect one environment step ───────────────────────
        if step < start_training:
            # Random warm-up — synchronous
            raw_action = env.action_space.sample()
            obs_next, reward, done, info = env.step(raw_action)

            # Convert to CUDA tensors (one small transfer per step)
            obs_flat_t  = torch.from_numpy(obs[-1].astype(np.float32)).to(dev)
            next_obs_t  = torch.from_numpy(obs_next[-1].astype(np.float32)).to(dev)
            a_flat_np   = raw_action.reshape(-1).astype(np.float32)
            if len(a_flat_np) != flat_action_dim:
                padded = np.zeros(flat_action_dim, dtype=np.float32)
                padded[:len(a_flat_np)] = a_flat_np
                a_flat_np = padded
            a_flat_t = torch.from_numpy(a_flat_np).to(dev)

            mask = 1.0 if (not done or info.get("TimeLimit.truncated", False)) else 0.0
            replay_buffer.insert(
                obs_flat_t, a_flat_t, reward, next_obs_t, float(done), mask
            )

        else:
            # OTF action selection — numpy→CUDA for env obs, CUDA→numpy for env action
            obs_flat_t  = torch.from_numpy(obs[-1].astype(np.float32)).to(dev)
            obs_multi_t = torch.from_numpy(obs[:n_obs_steps].astype(np.float32)).to(dev)

            with torch.no_grad():
                a_star_t = learner._sample_otf_action(
                    obs_flat_t.unsqueeze(0), obs_multi_t.unsqueeze(0),
                    use_target=True,
                ).squeeze(0)  # (flat_action_dim,) CUDA tensor

            # Convert action to numpy for env stepping
            a_star_np = a_star_t.cpu().numpy()
            action_chunk = a_star_np.reshape(n_action_steps, action_dim)
            env_action = action_chunk
            if abs_action and rotation_transformer is not None:
                env_action = undo_transform_action(
                    action_chunk[np.newaxis], rotation_transformer
                )[0]

            # Submit env step to background thread
            env_future = executor.submit(env.step, env_action)

            # ── Training updates while env is stepping ─────────
            if replay_buffer.size >= batch_size:
                # Sample — both return CUDA tensors, zero CPU involvement
                online_batch  = replay_buffer.sample(online_bs)
                offline_batch = offline_dataset.sample(offline_bs)

                # In-place fill pre-allocated CUDA tensors
                for k in ("observations", "actions", "rewards", "next_observations", "dones", "masks"):
                    combined[k][:online_bs] = online_batch[k]
                    combined[k][online_bs:] = offline_batch[k]

                # Build next_obs_multistep: tile on CUDA
                next_obs_single = combined["next_observations"]  # (B, Do)
                combined_next_multi[:] = next_obs_single.unsqueeze(1)
                combined["next_obs_multistep"] = combined_next_multi

                metrics = learner.update(combined)

                if step % 1000 == 0:
                    metrics_str = " | ".join(
                        f"{k}={v:.4f}" for k, v in metrics.items()
                    )
                    tqdm.tqdm.write(f"  [step {step}] {metrics_str}")
                    all_metrics.append({"step": step, **metrics})

            # ── Collect env result ─────────────────────────────
            obs_next, reward, done, info = env_future.result()

            next_obs_t = torch.from_numpy(obs_next[-1].astype(np.float32)).to(dev)
            mask = 1.0 if (not done or info.get("TimeLimit.truncated", False)) else 0.0
            replay_buffer.insert(
                obs_flat_t, a_star_t, reward, next_obs_t, float(done), mask
            )

        episode_return += reward
        episode_step   += 1
        obs = obs_next

        if done:
            episode_count += 1
            pbar.set_postfix(
                ep=episode_count,
                ret=f"{episode_return:.2f}",
                buf=replay_buffer.size,
            )
            obs = env.reset()
            episode_return = 0.0
            episode_step   = 0

        # ── Periodic evaluation ────────────────────────────────
        if step > 0 and step % eval_interval == 0:
            tqdm.tqdm.write(f"\n  Running evaluation at step {step} …")
            try:
                mean_score, runner_log = run_eval(cfg, learner, output_dir, device)
                tqdm.tqdm.write(f"  ✓ Eval mean_score = {mean_score:.4f}\n")
                all_metrics.append({
                    "step": step,
                    "eval_mean_score": mean_score,
                })
            except Exception as e:
                tqdm.tqdm.write(f"  ✗ Eval failed: {e}\n")

        # ── Periodic checkpoint ────────────────────────────────
        if step > 0 and step % eval_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:07d}.ckpt")
            learner.save(ckpt_path, base_checkpoint, expo_cfg)
            tqdm.tqdm.write(f"  Saved checkpoint: {ckpt_path}")

    # ── Cleanup ────────────────────────────────────────────────────
    executor.shutdown(wait=False)

    # ── Final save ─────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "latest.ckpt")
    learner.save(final_path, base_checkpoint, expo_cfg)
    print(f"\nFinal checkpoint saved: {final_path}")

    log_path = os.path.join(output_dir, "train_log.json")
    json.dump(all_metrics, open(log_path, "w"), indent=2)
    print(f"Training log saved: {log_path}")


if __name__ == "__main__":
    main()