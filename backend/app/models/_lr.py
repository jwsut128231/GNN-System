"""Shared LR scheduler factory.

User directive 2026-04-25: prefer basic exponential decay over adaptive/cosine
schedulers to produce smooth, monotonic training loss curves.
"""
import torch

DEFAULT_LR_GAMMA = 0.95


def build_scheduler(optimizer):
    """Return ExponentialLR with DEFAULT_LR_GAMMA per-epoch decay."""
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DEFAULT_LR_GAMMA)
