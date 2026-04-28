"""
CUDA-resident circular replay buffer for online EXPO transitions.

All storage and sampling happens on GPU — zero CPU↔GPU copies in the
training hot path.  The only CPU→GPU transfer is the single-transition
`.insert()` call after each environment step.
"""

import torch


class ReplayBuffer:
    """Circular replay buffer backed by CUDA tensors."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device

        self.observations      = torch.zeros(capacity, obs_dim, device=device)
        self.actions           = torch.zeros(capacity, action_dim, device=device)
        self.rewards           = torch.zeros(capacity, device=device)
        self.next_observations = torch.zeros(capacity, obs_dim, device=device)
        self.dones             = torch.zeros(capacity, device=device)
        self.masks             = torch.zeros(capacity, device=device)

    def insert(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: float,
        mask: float,
    ):
        """Insert a single transition. Inputs must be 1-D CUDA tensors (or scalars)."""
        i = self.ptr
        self.observations[i]      = obs
        self.actions[i]           = action
        self.next_observations[i] = next_obs
        self.rewards[i]           = reward
        self.dones[i]             = done
        self.masks[i]             = mask
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """Return a dict of CUDA tensors, already on self.device."""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "observations":      self.observations[idx],
            "actions":           self.actions[idx],
            "rewards":           self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "dones":             self.dones[idx],
            "masks":             self.masks[idx],
        }
