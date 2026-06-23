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

    def _sample_trajectories(self):
        """Sample expert and model trajectory batches (no reward computation)."""
        n_e = min(self.batch_size_expert, len(self.expert_trajectories))
        exp_idx = self.rng.choice(len(self.expert_trajectories), size=n_e, replace=False)
        expert_trajs = [self.expert_trajectories[i] for i in exp_idx]

        n_m = min(self.batch_size_model, len(self.trajectories))
        model_idx = self.rng.choice(len(self.trajectories), size=n_m, replace=False)
        model_trajs = [self.trajectories[i] for i in model_idx]
        return expert_trajs, model_trajs

    def _sample_returns(self, member):
        """Sample trajectories and compute differentiable whole-trajectory returns."""
        expert_trajs, model_trajs = self._sample_trajectories()
        expert_returns = th.stack([
            self._traj_sum_reward(member, traj) for traj in expert_trajs
        ])
        model_returns = th.stack([
            self._traj_sum_reward(member, traj) for traj in model_trajs
        ])
        return expert_returns, model_returns, expert_trajs, model_trajs

    def _reward_loss(self, member) -> th.Tensor:
        """Return the configured IRL loss while preserving historical formulas."""
        # ``maxent_corrected`` is the only loss with importance weights, so it is
        # the only one that benefits from (and uses) fragment-level partitioning.
        if self.loss_type == "maxent_corrected":
            return self._maxent_corrected_loss(member)

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

        # maxent
        return expert_term + th.logsumexp(model_returns, dim=0) - np.log(len(model_returns))

    def _maxent_corrected_loss(self, member) -> th.Tensor:
        """Importance-corrected MaxEnt NLL with optional fragment-level partition.

        With ``fragment_length=None`` each trajectory is a single fragment, which
        reproduces the historical whole-trajectory formula exactly. Shorter
        fragments shrink the variance of ``log q`` (which grows with horizon) and
        keep the importance-sampling effective sample size from collapsing.
        """
        expert_trajs, model_trajs = self._sample_trajectories()
        expert_returns = self._fragment_returns(member, expert_trajs)
        model_returns = self._fragment_returns(member, model_trajs)
        log_q = self._fragment_log_probs(model_trajs)
        corrected_logits = model_returns / self.temperature - log_q
        partition = th.logsumexp(corrected_logits, dim=0) - np.log(len(corrected_logits))
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

    def _traj_step_log_probs(self, traj: Trajectory) -> list:
        """Per-step policy log-probabilities, using stored values when available."""
        stored = [getattr(t, "log_policy_prob", None) for t in traj]
        if all(value is not None for value in stored):
            return [float(value) for value in stored]

        obs = np.asarray([t.observation for t in traj], dtype=np.float32)
        actions = np.asarray([t.action for t in traj])
        return [float(x) for x in policy_action_log_probs(self.agent, obs, actions)]

    def _traj_log_policy_prob(self, traj: Trajectory) -> float:
        return float(sum(self._traj_step_log_probs(traj)))

    def _traj_step_rewards(self, member, traj: Trajectory) -> th.Tensor:
        """Per-step rewards over a trajectory, preserving gradients. Shape (T,)."""
        obs = th.tensor(np.array([t.observation for t in traj]), dtype=th.float32)
        actions = th.tensor(np.array([t.action for t in traj]), dtype=th.float32)
        next_status = th.tensor(np.array([t.next_status for t in traj]), dtype=th.float32)
        done = th.tensor(np.array([float(t.done) for t in traj]), dtype=th.float32)
        return member(obs, actions, next_status, done)

    def _traj_sum_reward(self, member, traj: Trajectory) -> th.Tensor:
        """Sum per-step rewards over a trajectory, preserving gradients."""
        return self._traj_step_rewards(member, traj).sum()

    def _fragment_step(self, length: int) -> int:
        """Fragment size for a trajectory of ``length`` steps (None -> whole)."""
        if not self.fragment_length or self.fragment_length <= 0:
            return length
        return self.fragment_length

    def _fragment_returns(self, member, trajectories) -> th.Tensor:
        """Per-fragment differentiable reward sums across all trajectories."""
        fragments = []
        for traj in trajectories:
            per_step = self._traj_step_rewards(member, traj)
            length = per_step.shape[0]
            step = self._fragment_step(length)
            for i in range(0, length, step):
                fragments.append(per_step[i:i + step].sum())
        return th.stack(fragments)

    def _fragment_log_probs(self, trajectories) -> th.Tensor:
        """Per-fragment summed policy log-probabilities, aligned with returns."""
        out = []
        for traj in trajectories:
            log_probs = self._traj_step_log_probs(traj)
            length = len(log_probs)
            step = self._fragment_step(length)
            for i in range(0, length, step):
                out.append(float(sum(log_probs[i:i + step])))
        return th.tensor(out, dtype=th.float32)
