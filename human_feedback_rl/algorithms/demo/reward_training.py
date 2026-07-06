"""Optimization and agent-facing normalization of the learned reward."""

import time

import numpy as np
import torch as th


class RewardTrainingMixin:
    """Reward-model optimization methods used by ``DemoAlgorithm``."""

    @staticmethod
    def _grad_norm(module) -> float:
        squared_norms = [
            p.grad.pow(2).sum() for p in module.parameters() if p.grad is not None
        ]
        if not squared_norms:
            return 0.0
        return float(th.sqrt(th.stack(squared_norms).sum()))

    def _train_reward_model(self) -> None:
        if not self.trajectories:
            return

        self._maxent_corrected_steps = []
        t0 = time.perf_counter()

        def member_step(member, optimizer):
            member.train()
            norms = []
            for _ in range(self.gradient_steps_rew):
                loss = self._reward_loss(member)
                if not th.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite reward loss for loss_type={self.loss_type}: {loss.item()}"
                    )
                optimizer.zero_grad()
                loss.backward()
                grad_norm = self._grad_norm(member)
                if not np.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite reward gradient norm for loss_type={self.loss_type}."
                    )
                norms.append(grad_norm)
                optimizer.step()
            return norms

        all_norms = [norm for norms in self.train_reward_members(member_step) for norm in norms]

        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._log_reward_loss_diagnostics()
        self._log_maxent_corrected_step_diagnostics()
        self.logger.record("reward/grad_norm", float(np.mean(all_norms)), exclude="stdout")
        self.logger.record("reward/grad_norm_max", float(np.max(all_norms)), exclude="stdout")
        self.logger.record(
            "reward/weight_norm", self._param_norm(self.reward_model), exclude="stdout"
        )
        self.logger.record("time/train_reward_model", t_train)
        self.logger.record_sum("time/loggings", time.perf_counter() - t0)

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
