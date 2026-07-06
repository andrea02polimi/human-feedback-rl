"""Batched fragment/trajectory scoring for reward networks.

Replaces per-fragment Python loops of tiny forward passes with a single
forward over the concatenated transitions plus per-segment sums. The math is
unchanged; only the floating-point association order of the batched matmul
differs (observed deviation ~1e-8 on float32).

Transitions must have ``observation``, ``action``, ``next_status`` and
``done`` set (as produced by ``EnvBufferingWrapper``).
"""

from typing import List, Sequence, Tuple

import numpy as np
import torch as th

from .types import Trajectory

_CACHE_ATTR = "_stacked_transitions_cache"


def stacked_transitions(traj: Trajectory) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Stacked ``(obs, actions, next_status, done)`` float32 tensors for a trajectory.

    Memoized on the trajectory object itself (fragments in the preference
    dataset are scored many times across gradient steps, ensemble members and
    training rounds). The cache is invalidated if the trajectory length
    changes; the tensors carry no grad and are safe to reuse across backwards.
    """
    cached = getattr(traj, _CACHE_ATTR, None)
    if cached is not None and cached[0] == len(traj):
        return cached[1]

    tensors = (
        th.tensor(np.array([t.observation for t in traj]), dtype=th.float32),
        th.tensor(np.array([t.action for t in traj]), dtype=th.float32),
        th.tensor(np.array([t.next_status for t in traj]), dtype=th.float32),
        th.tensor(np.array([float(t.done) for t in traj]), dtype=th.float32),
    )
    try:
        setattr(traj, _CACHE_ATTR, (len(traj), tensors))
    except AttributeError:
        pass  # plain lists cannot hold attributes; skip caching
    return tensors


def per_step_rewards(net, trajectories: Sequence[Trajectory]) -> List[th.Tensor]:
    """Per-step rewards for each trajectory via one concatenated forward.

    Returns one (T_i,)-shaped tensor per trajectory, gradients preserved.
    """
    if not trajectories:
        return []
    lengths = [len(t) for t in trajectories]
    parts = [stacked_transitions(t) for t in trajectories]
    inputs = tuple(th.cat([p[i] for p in parts]) for i in range(4))
    out = net(*inputs)
    return list(th.split(out, lengths))


def fragment_sum_rewards(net, fragments: Sequence[Trajectory]) -> th.Tensor:
    """Per-fragment reward sums, shape (n_fragments,), gradients preserved."""
    return th.stack([steps.sum() for steps in per_step_rewards(net, fragments)])


def fragment_avg_rewards(net, fragments: Sequence[Trajectory]) -> th.Tensor:
    """Per-fragment mean rewards, shape (n_fragments,), gradients preserved."""
    lengths = th.tensor([len(f) for f in fragments], dtype=th.float32)
    return fragment_sum_rewards(net, fragments) / lengths
