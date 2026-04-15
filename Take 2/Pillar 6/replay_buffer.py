"""
Pillar 6 — Replay Buffer
=========================
Standard experience replay buffer for off-policy SAC training.
Stores (state, action, reward, next_state, done) transitions.
"""

import torch  # ALWAYS first (DLL conflict)
import numpy as np


class ReplayBuffer:
    """
    Fixed-size circular replay buffer storing numpy arrays.
    Samples are returned as PyTorch tensors on the requested device.
    """

    def __init__(self, state_dim, action_dim, max_size=1_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states      = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions      = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards      = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states  = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones        = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Store a single transition."""
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device="cpu"):
        """Sample a random mini-batch and return as tensors."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx],      dtype=torch.float32, device=device),
            torch.tensor(self.actions[idx],      dtype=torch.float32, device=device),
            torch.tensor(self.rewards[idx],      dtype=torch.float32, device=device),
            torch.tensor(self.next_states[idx],  dtype=torch.float32, device=device),
            torch.tensor(self.dones[idx],        dtype=torch.float32, device=device),
        )

    def __len__(self):
        return self.size
