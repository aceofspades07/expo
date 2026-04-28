"""
EXPOLearner — EXPO algorithm adapted for a frozen diffusion-policy base.

This module implements the full EXPO update loop:
  - G=20 critic updates per environment step (UTD ratio)
  - Edit policy RL + entropy update (1 per env step)
  - Entropy temperature α auto-tuning
  - On-The-Fly (OTF) action selection for both rollout and TD backup

The base diffusion policy is **never** updated here. It is loaded
from a pre-trained checkpoint and kept completely frozen.

The edit produces:  a_edited = a_base + β · tanh(δ)
where δ ~ π_edit(·|s, a_base).  (Smooth scaling, not hard clipping.)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffusion_policy.expo.networks import (
    GaussianEditPolicy,
    EnsembleCritic,
)


class EXPOLearner:
    """
    EXPO learner that wraps a frozen diffusion policy base with a trainable
    Gaussian edit policy and Q-ensemble critic.

    The diffusion policy interface:
        obs_dict = {"obs": tensor (B, To, Do)}
        result   = policy.predict_action(obs_dict)
        action   = result["action"]      # (B, Ta, Da)
        full     = result["action_pred"] # (B, T, Da)  (full horizon)

    For the edit policy and Q-ensemble, we flatten:
        obs_flat    = obs[:, -1, :]   →  (B, Do)       single-step obs
        action_flat = action.reshape(B, -1)  →  (B, Ta*Da)  flat action chunk
    """

    def __init__(
        self,
        base_policy: nn.Module,
        obs_dim: int,            # Do = 23
        action_dim: int,         # Da = 10
        n_obs_steps: int,        # To = 2
        n_action_steps: int,     # Ta = 8
        device: torch.device,
        # EXPO hyperparameters (paper Table 1)
        beta: float = 0.05,      # edit scale bound
        N: int = 8,              # action candidates
        utd_ratio: int = 20,     # critic updates per env step
        gamma: float = 0.99,     # discount
        tau: float = 0.005,      # Polyak coefficient
        lr: float = 3e-4,        # learning rate for all components
        batch_size: int = 256,
        num_q_networks: int = 10,
        num_min_q: int = 2,      # subsample min (paper Table 1)
        hidden_dim: int = 256,
        edit_dropout: float = 0.0,  # 0.1 for Adroit, 0.0 for Robomimic
    ):
        self.device = device
        self.beta = beta
        self.N = N
        self.utd_ratio = utd_ratio
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_min_q = num_min_q
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # Flat dimensions for edit policy / critic
        self.flat_action_dim = action_dim * n_action_steps  # 80

        # ── Frozen base policy ──────────────────────────────────────
        self.base_policy = base_policy
        self.base_policy.eval()
        for p in self.base_policy.parameters():
            p.requires_grad_(False)

        # ── Edit policy ─────────────────────────────────────────────
        self.edit_policy = GaussianEditPolicy(
            obs_dim=obs_dim,
            action_dim=self.flat_action_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            dropout=edit_dropout,
        ).to(device)

        # ── Critic ensemble ─────────────────────────────────────────
        self.critic = EnsembleCritic(
            obs_dim=obs_dim,
            action_dim=self.flat_action_dim,
            num_networks=num_q_networks,
            hidden_dim=hidden_dim,
        ).to(device)

        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # ── Entropy temperature (learnable) ─────────────────────────
        self.log_alpha = nn.Parameter(
            torch.tensor(0.0, device=device, dtype=torch.float32)
        )
        # SAC equilibrium: log_prob + target_entropy ≈ 0.
        # The edit policy starts with E[log_prob] ≈ -55, so for near-
        # equilibrium at init we need target_entropy ≈ +55.  We use
        # 0.65 * flat_action_dim = 52, giving a small initial gradient
        # that lets alpha adapt slowly rather than plummeting.
        self.target_entropy = 0.65 * self.flat_action_dim   # ≈ 52 for Square
        self.min_alpha = 0.2  # floor: always maintain meaningful exploration

        # ── Optimizers ──────────────────────────────────────────────
        self.edit_optimizer = torch.optim.Adam(
            self.edit_policy.parameters(), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr
        )
        # Alpha uses a slower learning rate for gradual exploration→exploitation
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

    @torch.no_grad()
    def _get_base_action_candidates_batched(
        self, obs_tensor: torch.Tensor, N: int
    ) -> torch.Tensor:
        """
        Sample N action candidates per state in a SINGLE batched forward pass.

        Instead of calling predict_action N times sequentially, we tile the
        observations to (B*N, To, Do) and run one large batch through the
        DDPM chain. Each sample gets independent noise from torch.randn
        inside the diffusion model, so all N candidates are different.

        Args:
            obs_tensor: (B, To, Do)
            N: number of candidates

        Returns:
            (B, N, flat_action_dim)
        """
        B, To, Do = obs_tensor.shape
        # Tile: (B, To, Do) -> (B*N, To, Do)
        obs_tiled = (
            obs_tensor.unsqueeze(1)
            .expand(-1, N, -1, -1)
            .reshape(B * N, To, Do)
        )
        obs_dict = {"obs": obs_tiled}
        # PERF: autocast to float16 for frozen base policy inference
        with torch.amp.autocast('cuda', dtype=torch.float16):
            result = self.base_policy.predict_action(obs_dict)
        action = result["action"].float()  # (B*N, Ta, Da) back to float32
        return action.reshape(B, N, -1)  # (B, N, flat_action_dim)

    def _sample_otf_action(
        self,
        obs_flat: torch.Tensor,
        obs_multistep: torch.Tensor,
        use_target: bool = True,
    ) -> torch.Tensor:
        """
        On-The-Fly action selection: sample N base actions, generate N edits,
        argmax Q over 2N candidates.

        Args:
            obs_flat:      (B, Do)    — single-step obs for edit policy / critic
            obs_multistep: (B, To, Do) — multi-step obs for base diffusion policy
            use_target:    use target critic (True for rollout & TD backup)

        Returns:
            (B, flat_action_dim) — best action (flat)
        """
        critic_fn = self.target_critic if use_target else self.critic
        B = obs_flat.shape[0]

        # 1. Sample N base actions from diffusion policy (frozen, no grad)
        with torch.no_grad():
            base_actions = self._get_base_action_candidates_batched(
                obs_multistep, self.N
            )  # (B, N, flat_action_dim)

        # 2. Generate N edits from edit policy
        obs_rep = obs_flat.unsqueeze(1).expand(-1, self.N, -1)  # (B, N, Do)
        with torch.no_grad():
            delta_dist = self.edit_policy(obs_rep, base_actions)
            delta = delta_dist.rsample()  # (B, N, flat_action_dim), already tanh'd
        edited_actions = base_actions + self.beta * delta  # smooth scaling

        # 3. Pool 2N candidates
        all_candidates = torch.cat(
            [base_actions, edited_actions], dim=1
        )  # (B, 2N, flat_action_dim)

        # 4. Score with full pessimistic min over ALL ensemble members.
        obs_rep2 = obs_flat.unsqueeze(1).expand(
            -1, 2 * self.N, -1
        )  # (B, 2N, Do)
        q_vals = critic_fn.min(
            obs_rep2.reshape(B * 2 * self.N, -1),
            all_candidates.reshape(B * 2 * self.N, -1),
        )  # (B*2N,)
        q_vals = q_vals.reshape(B, 2 * self.N)

        # 5. Conservative selection: only accept a candidate if it beats
        #    the median base Q-value.  This prevents the critic from
        #    overriding the near-optimal base policy until it's confident.
        q_base   = q_vals[:, :self.N]          # (B, N) — scores for base actions
        q_all    = q_vals                       # (B, 2N) — all scores

        # Best overall candidate
        best_idx_all = q_all.argmax(dim=1)      # (B,) — index into 2N
        # Best base-only candidate (safe fallback)
        best_idx_base = q_base.argmax(dim=1)    # (B,) — index into N

        # Threshold: median of base Q-values (robust to outliers)
        base_median = q_base.median(dim=1).values  # (B,)

        # Accept the overall best ONLY if it beats the base median
        best_q = q_all[torch.arange(B, device=self.device), best_idx_all]
        use_overall = best_q > base_median  # (B,) bool

        # Gather actions: use best overall where confident, else best base
        a_overall = all_candidates[
            torch.arange(B, device=self.device), best_idx_all
        ]
        a_base_best = base_actions[
            torch.arange(B, device=self.device), best_idx_base
        ]
        a_star = torch.where(use_overall.unsqueeze(-1), a_overall, a_base_best)
        return a_star

    def sample_action(
        self, obs_np: np.ndarray, obs_multistep_np: np.ndarray
    ) -> np.ndarray:
        """
        Used during online rollout collection.

        Args:
            obs_np:           (Do,)  — latest single-step observation
            obs_multistep_np: (To, Do) — multi-step observation window

        Returns:
            (flat_action_dim,) — OTF-selected flat action
        """
        obs_flat = (
            torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
        )
        obs_multi = (
            torch.from_numpy(obs_multistep_np)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            a_star = self._sample_otf_action(
                obs_flat, obs_multi, use_target=True
            )
        return a_star.squeeze(0).cpu().numpy()

    # ═══════════════════════════════════════════════════════════════
    #  EXPO Update (Algorithm 1 from paper)
    # ═══════════════════════════════════════════════════════════════
    def update(self, batch: dict) -> dict:
        """
        Full EXPO update step.

        Args:
            batch: dict of CUDA tensors with keys:
                "observations"       (batch_size, Do)
                "actions"            (batch_size, flat_action_dim)
                "rewards"            (batch_size,)
                "next_observations"  (batch_size, Do)
                "masks"              (batch_size,)
                "next_obs_multistep" (batch_size, To, Do)

        Returns:
            dict of scalar metrics for logging
        """
        # All tensors are already on self.device — zero conversion overhead
        s            = batch["observations"]
        a            = batch["actions"]
        r            = batch["rewards"]
        s_next       = batch["next_observations"]
        mask         = batch["masks"]
        s_next_multi = batch["next_obs_multistep"]

        metrics = {}

        # ── Precompute OTF target ONCE (same batch for all G steps) ──
        # We only precompute a_star_next because the diffusion UNet is slow.
        # q_target must be evaluated INSIDE the loop to maintain in-training 
        # target diversity (REDQ randomly subsamples critics per gradient step).
        with torch.no_grad():
            a_star_next = self._sample_otf_action(
                s_next, s_next_multi, use_target=True
            )

        # ── UTD loop: G=20 critic updates ──────────────────────────
        # Q-value bounds: reward ∈ [0,1], gamma=0.99 → Q_max = 1/(1-γ) = 100
        q_max = 1.0 / (1.0 - self.gamma)
        for g in range(self.utd_ratio):
            with torch.no_grad():
                q_target = r + self.gamma * mask * self.target_critic.subsample_min(
                    s_next, a_star_next, num_min=self.num_min_q
                )
                q_target = q_target.clamp(0.0, q_max)
            # Critic loss: MSE over all 10 networks
            q_preds = self.critic.all(s, a)
            critic_loss = (
                sum(F.mse_loss(q, q_target) for q in q_preds)
                / self.critic.num_networks
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Polyak update target critic
            with torch.no_grad():
                for p, p_tgt in zip(
                    self.critic.parameters(), self.target_critic.parameters()
                ):
                    p_tgt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        metrics["critic_loss"] = critic_loss.item()

        # ── Edit policy update: RL + entropy ───────────────────────
        # Sample base actions for the current states (frozen, no grad)
        with torch.no_grad():
            # Use the actions from the batch as base actions for edit training
            # This is equivalent to sampling from the base policy, but more
            # efficient since the batch actions were produced by OTF selection
            base_a = a  # (B, flat_action_dim)

        delta_dist = self.edit_policy(s, base_a)
        delta, log_prob = delta_dist.rsample_and_logprob()
        log_prob = log_prob.sum(-1)  # (B,)
        a_edited = base_a + self.beta * delta  # smooth scaling

        alpha = self.log_alpha.exp().detach()

        # Use full pessimistic min for edit policy — prevents the edit
        # from chasing overestimated Q-directions
        q_edited = self.critic.min(
            s, a_edited
        )

        edit_loss = -(q_edited - alpha * log_prob).mean()

        self.edit_optimizer.zero_grad()
        edit_loss.backward()
        self.edit_optimizer.step()

        metrics["edit_loss"] = edit_loss.item()
        metrics["alpha"] = alpha.item()

        # ── Alpha (entropy temperature) update ─────────────────────
        with torch.no_grad():
            delta_dist_detach = self.edit_policy(s, base_a)
            delta_detach, log_prob_detach = delta_dist_detach.rsample_and_logprob()
            log_prob_detach = log_prob_detach.sum(-1)

        alpha_loss = -(
            self.log_alpha * (log_prob_detach + self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Clamp alpha to minimum floor — never stop exploring entirely
        with torch.no_grad():
            self.log_alpha.data.clamp_(min=float(np.log(self.min_alpha)))

        metrics["alpha_loss"] = alpha_loss.item()
        metrics["mean_q"] = q_edited.mean().item()

        return metrics

    # ═══════════════════════════════════════════════════════════════
    #  Save / Load (edit components only — base is separate)
    # ═══════════════════════════════════════════════════════════════
    def save(self, path: str, base_checkpoint_path: str, cfg: dict):
        """Save EXPO-specific components (not the base policy)."""
        torch.save(
            {
                "edit_policy": self.edit_policy.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "log_alpha": self.log_alpha.data,
                "edit_optimizer": self.edit_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "base_checkpoint_path": base_checkpoint_path,
                "cfg": cfg,
            },
            path,
        )

    def load(self, path: str):
        """Load EXPO-specific components."""
        data = torch.load(path, map_location=self.device)
        self.edit_policy.load_state_dict(data["edit_policy"])
        self.critic.load_state_dict(data["critic"])
        self.target_critic.load_state_dict(data["target_critic"])
        self.log_alpha.data = data["log_alpha"].to(self.device)
        self.edit_optimizer.load_state_dict(data["edit_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(data["alpha_optimizer"])
        return data.get("cfg", {}), data.get("base_checkpoint_path", "")
