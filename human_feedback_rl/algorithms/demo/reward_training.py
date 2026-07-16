"""Optimization and agent-facing normalization of the learned reward."""

import numpy as np
import torch as th


class RewardTrainingMixin:
    """Reward-model optimization helpers used by ``HybridAlgorithm``."""

    @staticmethod
    def _grad_norm(module) -> float:
        squared_norms = [
            p.grad.pow(2).sum() for p in module.parameters() if p.grad is not None
        ]
        if not squared_norms:
            return 0.0
        return float(th.sqrt(th.stack(squared_norms).sum()))

    @staticmethod
    def _param_norm(module) -> float:
        return float(th.sqrt(sum(p.detach().pow(2).sum() for p in module.parameters())))

    def _update_agent_reward_normalization(self) -> None:
        """Fit the agent-only reward transform on the current rollout."""
        if not self.normalize_agent_reward or not self.trajectories:
            return

        transitions = [transition for trajectory in self.trajectories for transition in trajectory]
        if not transitions:
            return

        obs = np.asarray([t.observation for t in transitions], dtype=np.float32)
        actions = np.asarray([t.action for t in transitions], dtype=np.float32)
        statuses = np.asarray([t.next_status for t in transitions], dtype=np.float32)
        dones = np.asarray([float(t.done) for t in transitions], dtype=np.float32)

        raw_rewards = self.reward_model.predict_unnormalized(obs, actions, statuses, dones)
        if not np.isfinite(raw_rewards).all():
            raise FloatingPointError("Non-finite raw rewards while fitting agent normalization.")

        raw_mean = float(np.mean(raw_rewards))
        raw_std = float(np.std(raw_rewards))
        safe_std = raw_std if raw_std > 1e-8 else 1.0
        self.reward_model.set_mean(raw_mean)
        self.reward_model.set_std(safe_std)

        normalized_rewards = self.reward_model.predict(obs, actions, statuses, dones)
        if not np.isfinite(normalized_rewards).all():
            raise FloatingPointError("Non-finite normalized rewards for agent training.")

        self.logger.record("reward/normalization_raw_mean", raw_mean, exclude="stdout")
        self.logger.record("reward/normalization_raw_std", raw_std, exclude="stdout")
        self.logger.record(
            "reward/normalization_applied_mean",
            self.reward_model.normalization_mean,
            exclude="stdout",
        )
        self.logger.record(
            "reward/normalization_applied_std",
            self.reward_model.normalization_std,
            exclude="stdout",
        )
        self.logger.record(
            "reward/normalization_output_mean",
            float(np.mean(normalized_rewards)),
            exclude="stdout",
        )
        self.logger.record(
            "reward/normalization_output_std",
            float(np.std(normalized_rewards)),
            exclude="stdout",
        )
