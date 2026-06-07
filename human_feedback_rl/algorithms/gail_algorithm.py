"""
GailAlgorithm: GAIL-style imitation learning with fixed offline expert demonstrations.

Replaces the MaxEnt IRL loss of DemoAlgorithm with a binary cross-entropy discriminator:

    L = -mean_{(s,a)~expert}[log D(s,a)] - mean_{(s,a)~agent}[log(1 - D(s,a))]

The reward passed to PPO is the raw logit of the discriminator:

    r(s,a) = logit(D(s,a)) = log(D / (1-D))

When π_agent ≈ π_expert the discriminator outputs D → 0.5 and r → 0 (not inverted),
which prevents the behavioral inversion observed with MaxEnt IRL.
"""

from typing import List, Optional, Union, Callable

import numpy as np
import torch as th
import torch.nn.functional as F

from .demo_algorithm import DemoAlgorithm
from ..common.types import Trajectory


class GailAlgorithm(DemoAlgorithm):
    """GAIL discriminator trained with BCE instead of MaxEnt IRL."""

    # ------------------------------------------------------------------
    # Loss override
    # ------------------------------------------------------------------

    def _compute_reward_loss(self, member) -> th.Tensor:
        return self._gail_loss(member)

    def _gail_loss(self, member) -> th.Tensor:
        """Binary cross-entropy discriminator loss per transition.

        Expert transitions → label 1 (look like expert).
        Agent transitions  → label 0 (look like agent).
        """
        # ── Expert transitions ───────────────────────────────────────────
        obs_e, act_e, ns_e, done_e = self._batch_transitions(
            self.expert_trajectories, self.batch_size_expert
        )
        logits_e = member(obs_e, act_e, ns_e, done_e)
        loss_e = F.binary_cross_entropy_with_logits(
            logits_e, th.ones_like(logits_e)
        )

        # ── Agent transitions (anchor + recent) ──────────────────────────
        model_pool = list(self.model_buffer)

        if self.anchor_buffer and model_pool:
            n_anc = max(1, int(self.anchor_frac * self.batch_size_model))
            n_rec = self.batch_size_model - n_anc
            obs_a, act_a, ns_a, done_a = self._batch_transitions_mixed(
                self.anchor_buffer, n_anc, model_pool, n_rec
            )
        else:
            obs_a, act_a, ns_a, done_a = self._batch_transitions(
                self.trajectories, self.batch_size_model
            )

        logits_a = member(obs_a, act_a, ns_a, done_a)
        loss_a = F.binary_cross_entropy_with_logits(
            logits_a, th.zeros_like(logits_a)
        )

        return (loss_e + loss_a) / 2

    # ------------------------------------------------------------------
    # Skip mean normalization — logit reward is already centered at 0
    # ------------------------------------------------------------------

    def before_agent_training(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Transition sampling helpers
    # ------------------------------------------------------------------

    def _batch_transitions(self, trajectories: list, n: int):
        """Sample n transitions uniformly from a list of trajectories."""
        all_t = [t for traj in trajectories for t in traj]
        n = min(n, len(all_t))
        replace = len(all_t) < n
        idx = self.rng.choice(len(all_t), size=n, replace=replace)
        return self._transitions_to_tensors([all_t[i] for i in idx])

    def _batch_transitions_mixed(self, pool1: list, n1: int, pool2: list, n2: int):
        """Sample n1 transitions from pool1 and n2 from pool2, concatenate."""
        t1 = self._batch_transitions(pool1, n1)
        t2 = self._batch_transitions(pool2, n2)
        return tuple(th.cat([a, b]) for a, b in zip(t1, t2))

    @staticmethod
    def _transitions_to_tensors(transitions):
        obs    = th.tensor(np.array([t.observation  for t in transitions]), dtype=th.float32)
        acts   = th.tensor(np.array([t.action       for t in transitions]), dtype=th.float32)
        ns     = th.tensor(np.array([t.next_status  for t in transitions]), dtype=th.float32)
        done   = th.tensor(np.array([float(t.done)  for t in transitions]), dtype=th.float32)
        return obs, acts, ns, done
