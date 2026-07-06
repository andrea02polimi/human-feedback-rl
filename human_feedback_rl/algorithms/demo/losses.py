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


# ---------------------------------------------------------------------------
# Pure loss functions (tensor in, scalar tensor out)
# ---------------------------------------------------------------------------

def maxent_loss(expert_returns: th.Tensor, model_returns: th.Tensor) -> th.Tensor:
    """Historical MaxEnt surrogate: model-only partition estimate."""
    return -expert_returns.mean() + th.logsumexp(model_returns, dim=0) - np.log(len(model_returns))


def maxent2_loss(expert_returns: th.Tensor, model_returns: th.Tensor) -> th.Tensor:
    """Historical MaxEnt surrogate: expert+model partition estimate."""
    all_returns = th.cat([model_returns, expert_returns], dim=0)
    return -expert_returns.mean() + th.logsumexp(all_returns, dim=0) - np.log(len(all_returns))


def demo_loss(expert_returns: th.Tensor, model_returns: th.Tensor) -> th.Tensor:
    """Historical difference-of-means loss."""
    return -expert_returns.mean() + model_returns.mean()


def demo_corrected_loss(margins: th.Tensor, temperature: float) -> th.Tensor:
    """Bounded ranking loss on per-pair expert-minus-model mean-reward margins."""
    return F.softplus(-margins / temperature).mean()


def maxent_corrected_partition(corrected_logits: th.Tensor) -> th.Tensor:
    """Importance-corrected log-partition estimate over fragment logits R/tau - log q."""
    return th.logsumexp(corrected_logits, dim=0) - np.log(len(corrected_logits))


class RewardLossMixin:
    """Sampling and loss orchestration used by :class:`DemoAlgorithm`.

    The loss formulas themselves are the module-level pure functions above;
    this mixin samples trajectory batches, computes differentiable returns,
    and dispatches on ``self.loss_type``.
    """

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

        if self.loss_type in ("demo", "demo_loss"):
            return demo_loss(expert_returns, model_returns)

        if self.loss_type == "demo_corrected":
            margins = self._demo_corrected_margins(
                expert_returns, model_returns, expert_trajs, model_trajs
            )
            return demo_corrected_loss(margins, self.temperature)

        if self.loss_type == "maxent_2":
            return maxent2_loss(expert_returns, model_returns)

        return maxent_loss(expert_returns, model_returns)

    def _maxent_corrected_loss(self, member) -> th.Tensor:
        """Importance-corrected MaxEnt NLL with optional fragment-level partition.

        With ``fragment_length=None`` each trajectory is a single fragment, which
        reproduces the historical whole-trajectory formula exactly (verified
        bit-for-bit) and is the consistent importance-sampling estimator.

        CAVEAT — ``fragment_length>0`` changes the objective and is NOT proven
        correct. Fragments are consecutive chunks of one rollout, not i.i.d. draws
        from a fixed proposal, so the importance-sampling consistency that
        justifies ``maxent_corrected`` does not transfer to the fragment level.
        It becomes a *local* (windowed) MaxEnt objective in the spirit of
        GCL/AIRL: a heuristic for shrinking the ``log q`` variance, whose benefit
        is unverified. Treat it as experimental, not as "the correct loss".
        """
        expert_trajs, model_trajs = self._sample_trajectories()
        expert_returns = self._fragment_returns(member, expert_trajs)
        model_returns = self._fragment_returns(member, model_trajs)
        log_q = self._fragment_log_probs(model_trajs)
        scaled_returns = model_returns / self.temperature
        corrected_logits = scaled_returns - log_q
        partition = maxent_corrected_partition(corrected_logits)
        self._record_maxent_corrected_step(scaled_returns, log_q, corrected_logits)
        return -expert_returns.mean() / self.temperature + partition

    def _record_maxent_corrected_step(self, scaled_returns, log_q, logits) -> None:
        """Stash ESS and variance decomposition of the *actual* gradient sample.

        Lets us see which term inflates the partition-logit spread that controls
        the importance-sampling ESS: ``Var(R/τ)`` (reward scale) vs ``Var(log q)``
        (proposal/horizon), with their covariance. Logged (averaged) by the
        reward-training loop, so it reflects the gradient that was really applied.
        """
        if not hasattr(self, "_maxent_corrected_steps"):
            self._maxent_corrected_steps = []
        with th.no_grad():
            n = logits.shape[0]
            weights = th.softmax(logits, dim=0)
            ess = float(1.0 / weights.pow(2).sum())
            top_k = th.topk(weights, k=min(5, n)).values
            var_scaled = float(scaled_returns.var(unbiased=False))
            var_log_q = float(log_q.var(unbiased=False))
            if n > 1:
                cov = float(
                    ((scaled_returns - scaled_returns.mean()) * (log_q - log_q.mean())).mean()
                )
            else:
                cov = 0.0
            self._maxent_corrected_steps.append({
                "ess": ess,
                "ess_fraction": ess / n,
                "n_fragments": float(n),
                "top1_softmax_weight": float(weights.max()),
                "top5_softmax_mass": float(top_k.sum()),
                "logit_var": float(logits.var(unbiased=False)),
                "var_scaled_return": var_scaled,
                "var_log_q": var_log_q,
                "cov_return_log_q": cov,
            })

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
