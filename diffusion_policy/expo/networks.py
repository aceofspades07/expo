"""
EXPO network components: TanhNormal distribution, Gaussian edit policy,
single Q-network, and ensemble critic.

All architectures follow the EXPO paper (Dong et al., 2025) exactly:
- Edit policy: 3 hidden layers, 256 dim, ReLU, TanhNormal output
- Q-network:   3 hidden layers, 256 dim, ReLU, scalar output
- Ensemble:    10 independent Q-networks, pessimistic min aggregation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ═══════════════════════════════════════════════════════════════════════════
#  TanhNormal — Squashed Gaussian distribution
# ═══════════════════════════════════════════════════════════════════════════
class TanhNormal:
    """
    Gaussian projected through tanh.  Provides reparameterized sampling
    and the correct log-probability under the change-of-variables formula.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self._normal = Normal(mean, std)

    def rsample(self) -> torch.Tensor:
        """Reparameterized sample: z ~ N(μ, σ²), return tanh(z)."""
        z = self._normal.rsample()
        return torch.tanh(z)

    def rsample_and_logprob(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob) avoiding numerical instability from clamping."""
        z = self._normal.rsample()
        y = torch.tanh(z)
        log_prob_z = self._normal.log_prob(z)
        # Jacobian correction: log(1 - tanh(z)^2)
        # Numerically stable formulation: 2 * log(2) - 2 * |z| - 2 * softplus(-2|z|)
        z_abs = z.abs()
        log_jacob = 2.0 * math.log(2.0) - 2.0 * z_abs - 2.0 * F.softplus(-2.0 * z_abs)
        log_prob_y = log_prob_z - log_jacob
        return y, log_prob_y

    @property
    def mean(self) -> torch.Tensor:
        return torch.tanh(self._normal.mean)


# ═══════════════════════════════════════════════════════════════════════════
#  Gaussian Edit Policy  π_edit(δ | s, a_base)
# ═══════════════════════════════════════════════════════════════════════════
class GaussianEditPolicy(nn.Module):
    """
    Gaussian edit policy π_edit(δ | s, a_base).

    Input:  (state, base_action) concatenated  →  (obs_dim + action_dim,)
    Output: TanhNormal distribution over residual δ.

    The final edited action is:  a_edited = a_base + β · tanh(δ)
    where δ ~ TanhNormal(mean, std), so the edit is smoothly bounded to [-β, β].

    Architecture: 3 hidden layers, 256 dim, ReLU (paper Section 9.2).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        input_dim = obs_dim + action_dim
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, states: torch.Tensor, base_actions: torch.Tensor
    ) -> TanhNormal:
        """
        Args:
            states:       (..., obs_dim)
            base_actions: (..., action_dim)
        Returns:
            TanhNormal distribution over δ with same leading dims.
        """
        leading_shape = states.shape[:-1]
        x = torch.cat(
            [
                states.reshape(-1, states.shape[-1]),
                base_actions.reshape(-1, base_actions.shape[-1]),
            ],
            dim=-1,
        )
        h = self.net(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5.0, 2.0)
        std = log_std.exp()
        return TanhNormal(
            mean.reshape(*leading_shape, -1),
            std.reshape(*leading_shape, -1),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Single Q(s, a) network
# ═══════════════════════════════════════════════════════════════════════════
class StateActionValue(nn.Module):
    """
    Single Q(s, a) network with DroQ regularization.
    3 hidden layers, 256 dim, LayerNorm + ReLU + Dropout → scalar output.
    LayerNorm + Dropout are critical for high-UTD stability (DroQ paper).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.01,
    ):
        super().__init__()
        dims = [obs_dim + action_dim] + [hidden_dim] * num_layers + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Returns Q-value scalar for each (s, a) pair. Shape: (B,)."""
        return self.net(torch.cat([states, actions], dim=-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
#  Ensemble Critic  (10 Q-networks, pessimistic aggregation)
# ═══════════════════════════════════════════════════════════════════════════
class EnsembleCritic(nn.Module):
    """
    10-member Q-function ensemble (paper Table 1).
    - .all(s, a)  → list of Q-values from each member
    - .min(s, a)  → min over all 10 (full pessimism)
    - .subsample_min(s, a, num_min=2) → min over random subset of 2
      (Num Min Q = 2, paper Table 1)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_networks: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_networks = num_networks
        self.networks = nn.ModuleList(
            [
                StateActionValue(obs_dim, action_dim, hidden_dim)
                for _ in range(num_networks)
            ]
        )

    def all(self, states: torch.Tensor, actions: torch.Tensor) -> list:
        """Returns list of Q-values from each network. Each element: (B,)."""
        return [net(states, actions) for net in self.networks]

    def min(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Pessimistic estimate: min over all ensemble members. Shape: (B,)."""
        qs = torch.stack(self.all(states, actions), dim=0)  # (num_networks, B)
        return qs.min(dim=0).values

    def subsample_min(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        num_min: int = 2,
    ) -> torch.Tensor:
        """
        RLPD-style random subsample minimum.
        Randomly pick `num_min` networks and return their min.
        Paper Table 1: Num Min Q = 2.
        Only evaluates the chosen networks (not all 10).
        """
        idx = torch.randperm(self.num_networks)[:num_min]
        qs = torch.stack(
            [self.networks[i](states, actions) for i in idx], dim=0
        )
        return qs.min(dim=0).values

