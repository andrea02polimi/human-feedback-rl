"""
GailAlgorithm: GAIL-style imitation learning with fixed offline expert demonstrations.

Replaces the MaxEnt IRL loss of DemoAlgorithm with a binary cross-entropy discriminator:

    max_D E_{(s,a)~agent}[log D(s,a)] + E_{(s,a)~expert}[log(1 - D(s,a))]

where D(s,a) is the probability that a transition came from the policy. The
policy maximizes the corresponding reward:

    r(s,a) = -log D(s,a)

This is the reward-equivalent form of minimizing E_pi[log D(s,a)] in the
original GAIL objective.
"""

from typing import List, Optional, Union, Callable

import numpy as np
import torch as th
import torch.nn.functional as F

from .demo_algorithm import DemoAlgorithm
from ..common.reward_nets import make_gail_discriminator_ensemble
from ..common.types import Trajectory


class GailAlgorithm(DemoAlgorithm):
    """GAIL discriminator trained with the original paper's label convention."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("reward_model_factory", make_gail_discriminator_ensemble)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Loss override
    # ------------------------------------------------------------------

    def _update_importance_weights(self) -> None:
        # The GAIL/AIRL discriminator loss does not use importance weights, so we
        # skip the policy snapshot and fusion-weight recomputation entirely.
        pass

    def _compute_reward_loss(self, member, eval_batch=None) -> th.Tensor:
        # GAIL samples its own transition batches from the current rollout, so the
        # trajectory-based ``eval_batch`` snapshot is not used here.
        return self._gail_loss(member)

    def _gail_loss(self, member) -> th.Tensor:
        """Binary cross-entropy discriminator loss per transition.

        D(s,a) is the probability that a transition came from the policy:
        expert transitions → label 0
        agent transitions  → label 1
        """
        # ── Expert transitions ───────────────────────────────────────────
        obs_e, act_e, ns_e, done_e = self._batch_transitions(
            self.expert_trajectories, self.batch_size_expert
        )
        logits_e = member(obs_e, act_e, ns_e, done_e)
        loss_e = F.binary_cross_entropy_with_logits(
            logits_e, th.zeros_like(logits_e)
        )

        # ── Agent transitions from the current policy rollout ────────────
        obs_a, act_a, ns_a, done_a = self._batch_transitions(
            self.trajectories, self.batch_size_model
        )

        logits_a = member(obs_a, act_a, ns_a, done_a)
        loss_a = F.binary_cross_entropy_with_logits(
            logits_a, th.ones_like(logits_a)
        )

        return loss_e + loss_a

    # ------------------------------------------------------------------
    # GAIL uses -log D(s,a) directly as the policy reward.
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

    @staticmethod
    def _transitions_to_tensors(transitions):
        obs    = th.tensor(np.array([t.observation  for t in transitions]), dtype=th.float32)
        acts   = th.tensor(np.array([t.action       for t in transitions]), dtype=th.float32)
        ns     = th.tensor(np.array([t.next_status  for t in transitions]), dtype=th.float32)
        done   = th.tensor(np.array([float(t.done)  for t in transitions]), dtype=th.float32)
        return obs, acts, ns, done
