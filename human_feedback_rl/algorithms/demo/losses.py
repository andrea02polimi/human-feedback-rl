"""Reward objectives and trajectory scoring for demonstration-based IRL."""

import numpy as np
import torch as th
import torch.nn.functional as F

from human_feedback_rl.common.trajectory_generators import policy_action_log_probs
from human_feedback_rl.common.types import Trajectory


VALID_LOSSES = (
    "maxent",
    "maxent_2",
    "demo",
    "demo_loss",
    "maxent_corrected",
    "demo_corrected",
)


class RewardLossMixin:
    """Loss computation methods used by :class:`DemoAlgorithm`."""

    def _sample_returns(self, member):
        """Sample trajectories and compute differentiable reward returns."""
        n_e = min(self.batch_size_expert, len(self.expert_trajectories))
        exp_idx = self.rng.choice(len(self.expert_trajectories), size=n_e, replace=False)
        expert_trajs = [self.expert_trajectories[i] for i in exp_idx]
        expert_returns = th.stack([
            self._traj_sum_reward(member, traj) for traj in expert_trajs
        ])

        n_m = min(self.batch_size_model, len(self.trajectories))
        model_idx = self.rng.choice(len(self.trajectories), size=n_m, replace=False)
        model_trajs = [self.trajectories[i] for i in model_idx]
        model_returns = th.stack([
            self._traj_sum_reward(member, traj) for traj in model_trajs
        ])
        return expert_returns, model_returns, expert_trajs, model_trajs

    def _reward_loss(self, member) -> th.Tensor:
        """Return the configured IRL loss while preserving historical formulas."""
        expert_returns, model_returns, expert_trajs, model_trajs = self._sample_returns(member)

        expert_term = -expert_returns.mean()
        if self.loss_type in ("demo", "demo_loss"):
            return expert_term + model_returns.mean()

        if self.loss_type == "demo_corrected":
            margins = self._demo_corrected_margins(
                expert_returns, model_returns, expert_trajs, model_trajs
            )
            return F.softplus(-margins / self.temperature).mean()

        if self.loss_type == "maxent_2":
            all_returns = th.cat([model_returns, expert_returns], dim=0)
            return expert_term + th.logsumexp(all_returns, dim=0) - np.log(len(all_returns))

        if self.loss_type == "maxent":
            return expert_term + th.logsumexp(model_returns, dim=0) - np.log(len(model_returns))

        log_q = th.tensor(
            [self._traj_log_policy_prob(traj) for traj in model_trajs], dtype=th.float32
        )
        corrected_logits = model_returns / self.temperature - log_q
        partition = th.logsumexp(corrected_logits, dim=0) - np.log(len(model_returns))
        return -expert_returns.mean() / self.temperature + partition

    @staticmethod
    def _demo_corrected_margins(expert_returns, model_returns, expert_trajs, model_trajs):
        n_pairs = min(len(expert_returns), len(model_returns))
        expert_scores = th.stack([
            expert_returns[i] / len(expert_trajs[i]) for i in range(n_pairs)
        ])
        model_scores = th.stack([
            model_returns[i] / len(model_trajs[i]) for i in range(n_pairs)
        ])
        return expert_scores - model_scores

    def _traj_log_policy_prob(self, traj: Trajectory) -> float:
        stored = [getattr(t, "log_policy_prob", None) for t in traj]
        if all(value is not None for value in stored):
            return float(sum(stored))

        obs = np.asarray([t.observation for t in traj], dtype=np.float32)
        actions = np.asarray([t.action for t in traj])
        return float(policy_action_log_probs(self.agent, obs, actions).sum())

    def _traj_sum_reward(self, member, traj: Trajectory) -> th.Tensor:
        """Sum per-step rewards over a trajectory, preserving gradients."""
        obs = th.tensor(np.array([t.observation for t in traj]), dtype=th.float32)
        actions = th.tensor(np.array([t.action for t in traj]), dtype=th.float32)
        next_status = th.tensor(np.array([t.next_status for t in traj]), dtype=th.float32)
        done = th.tensor(np.array([float(t.done) for t in traj]), dtype=th.float32)
        return member(obs, actions, next_status, done).sum()
