"""Reward objectives and trajectory scoring for demonstration-based IRL.

Two demonstration losses, both on differentiable whole-trajectory return sums:

    demo_1   difference of means: -E[expert] + E[model]
    demo_2   MaxEnt surrogate with the partition estimated over expert+model
"""

import numpy as np
import torch as th

from human_feedback_rl.common.batching import fragment_sum_rewards


VALID_LOSSES = ("demo_1", "demo_2")


# ---------------------------------------------------------------------------
# Pure loss functions (tensor in, scalar tensor out)
# ---------------------------------------------------------------------------

def demo_1_loss(expert_returns: th.Tensor, model_returns: th.Tensor) -> th.Tensor:
    """Difference of means: the experts should score above the rollout."""
    return -expert_returns.mean() + model_returns.mean()


def demo_2_loss(expert_returns: th.Tensor, model_returns: th.Tensor) -> th.Tensor:
    """MaxEnt surrogate, with the partition estimated over experts and rollout."""
    all_returns = th.cat([model_returns, expert_returns], dim=0)
    return -expert_returns.mean() + th.logsumexp(all_returns, dim=0) - np.log(len(all_returns))


class RewardLossMixin:
    """Sampling and loss orchestration used by :class:`HybridAlgorithm`.

    The loss formulas themselves are the module-level pure functions above;
    this mixin samples trajectory batches, computes differentiable returns,
    and dispatches on ``self.loss_type``.
    """

    def _sample_trajectories(self):
        """Sample expert and model trajectory batches (no reward computation)."""
        n_e = min(self.batch_size_expert, len(self.expert_trajectories))
        exp_idx = self._rng_train.choice(len(self.expert_trajectories), size=n_e, replace=False)
        expert_trajs = [self.expert_trajectories[i] for i in exp_idx]

        n_m = min(self.batch_size_model, len(self.trajectories))
        model_idx = self._rng_train.choice(len(self.trajectories), size=n_m, replace=False)
        model_trajs = [self.trajectories[i] for i in model_idx]
        return expert_trajs, model_trajs

    def _sample_returns(self, member):
        """Sample trajectories and compute differentiable whole-trajectory returns."""
        expert_trajs, model_trajs = self._sample_trajectories()
        expert_returns = fragment_sum_rewards(member, expert_trajs)
        model_returns = fragment_sum_rewards(member, model_trajs)
        return expert_returns, model_returns, expert_trajs, model_trajs

    def _reward_loss(self, member) -> th.Tensor:
        """Return the configured demonstration loss."""
        expert_returns, model_returns, _, _ = self._sample_returns(member)
        if self.loss_type == "demo_1":
            return demo_1_loss(expert_returns, model_returns)
        return demo_2_loss(expert_returns, model_returns)
