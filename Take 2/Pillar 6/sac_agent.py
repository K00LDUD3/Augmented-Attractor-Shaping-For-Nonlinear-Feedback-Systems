"""
Pillar 6 — Bare-Metal Soft Actor-Critic (SAC) Agent
=====================================================
Full PyTorch implementation of SAC with automatic entropy tuning.
No SB3 dependency — complete control over architecture and gradients.

Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
           Deep RL with a Stochastic Actor" (2018).

Architecture:
    Actor:  2x256 MLP -> mean, log_std -> TanhNormal squashing
    Critic: Twin Q-networks, each 2x256 MLP
    Alpha:  Automatic entropy coefficient tuning
"""

import torch  # ALWAYS first (DLL conflict)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import copy


# ---------------------------------------------------------------------------
# Network Architectures
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    """
    Stochastic policy network.
    Outputs mean and log_std of a Gaussian, then squashes through tanh.
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Sample an action using the reparameterisation trick.
        Returns: (tanh-squashed action, log_prob)
        """
        mean, log_std = self.forward(state)
        std  = log_std.exp()
        dist = Normal(mean, std)

        # Reparameterised sample
        x_t    = dist.rsample()
        action = torch.tanh(x_t)

        # Log-probability with tanh correction (Appendix C of SAC paper)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, state):
        """Return the deterministic (mean) action for evaluation."""
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class Critic(nn.Module):
    """Twin Q-network (two independent Q-functions)."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------
class SACAgent:
    """
    Soft Actor-Critic with automatic entropy tuning.

    Key hyperparameters:
        gamma:     Discount factor
        tau:       Soft target update coefficient
        alpha_lr:  Learning rate for the entropy temperature
        actor_lr:  Learning rate for the actor
        critic_lr: Learning rate for the critic
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        init_alpha=0.2,
        device="cpu",
    ):
        self.gamma  = gamma
        self.tau    = tau
        self.device = torch.device(device)
        self.action_dim = action_dim

        # --- Networks ---
        self.actor  = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # Freeze target parameters
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # --- Optimisers ---
        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # --- Automatic entropy tuning ---
        # Target entropy = -dim(A) (heuristic from Haarnoja et al.)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float32,
            device=self.device, requires_grad=True
        )
        self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        """
        Select an action given a numpy state vector.
        Returns a numpy action in [-1, 1].
        """
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if deterministic:
                action = self.actor.deterministic(state_t)
            else:
                action, _ = self.actor.sample(state_t)
            return action.cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size=256):
        """
        Perform one gradient step on actor, critic, and alpha.
        Returns a dict of training metrics.
        """
        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size, device=self.device
        )

        # ---- Critic Update ----
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            td_target = rewards + (1.0 - dones) * self.gamma * q_target

        q1_pred, q2_pred = self.critic(states, actions)
        critic_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- Actor Update ----
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ---- Alpha (Entropy Temperature) Update ----
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ---- Soft Target Update ----
        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss.item(),
            "alpha":       self.alpha.item(),
            "q1_mean":     q1_pred.mean().item(),
            "q2_mean":     q2_pred.mean().item(),
            "log_prob_mean": log_probs.mean().item(),
        }

    def _soft_update(self):
        """Polyak-average the target critic toward the online critic."""
        for p_target, p_online in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            p_target.data.mul_(1.0 - self.tau)
            p_target.data.add_(self.tau * p_online.data)

    def save(self, path):
        """Save all network weights and optimiser states."""
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "actor_optim":    self.actor_optim.state_dict(),
            "critic_optim":   self.critic_optim.state_dict(),
            "log_alpha":      self.log_alpha.detach().cpu(),
            "alpha_optim":    self.alpha_optim.state_dict(),
        }, path)

    def load(self, path, evaluate=False):
        """Load all network weights and optionally set to eval mode."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.alpha_optim.defaults['lr'])
        self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
        if evaluate:
            self.actor.eval()
            self.critic.eval()

    def get_params_dict(self):
        """Return all agent hyperparameters for experiment logging."""
        return {
            "agent_type": "SAC",
            "state_dim": self.actor.net[0].in_features,
            "action_dim": self.action_dim,
            "hidden_dim": self.actor.net[0].out_features,
            "gamma": self.gamma,
            "tau": self.tau,
            "actor_lr": self.actor_optim.defaults['lr'],
            "critic_lr": self.critic_optim.defaults['lr'],
            "alpha_lr": self.alpha_optim.defaults['lr'],
            "init_alpha": self.alpha.item(),
            "target_entropy": self.target_entropy,
            "actor_params": sum(p.numel() for p in self.actor.parameters()),
            "critic_params": sum(p.numel() for p in self.critic.parameters()),
        }
