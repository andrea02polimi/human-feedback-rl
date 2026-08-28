"""How well the learned reward ranks and separates trajectories."""

import numpy as np
import torch as th
from scipy.stats import pearsonr, spearmanr
from scipy.stats import pearsonr, spearmanr
from human_feedback_rl.common.batching import fragment_sum_rewards
from human_feedback_rl.common.types import Trajectory


class RewardValidationMixin:
    """Correlation, ranking and outcome-separation metrics."""

    def _diagnostic_returns(self, member):
        expert_trajs = self._even_subset(self.expert_trajectories, self.batch_size_expert)
        model_trajs = self._even_subset(self.trajectories, self.batch_size_model)
        expert_returns = fragment_sum_rewards(member, expert_trajs)
        model_returns = fragment_sum_rewards(member, model_trajs)
        return expert_returns, model_returns, expert_trajs, model_trajs

    @staticmethod
    def _even_subset(items, max_items: int):
        n_items = len(items)
        if max_items <= 0:
            max_items = 1
        if n_items <= max_items:
            return list(items)
        indices = np.linspace(0, n_items - 1, max_items, dtype=int)
        return [items[i] for i in indices]

    def _log_validation_snapshot(self, rollout_transitions, stage: str) -> None:
        rollout_prefix = f"reward_val/current_rollout/{stage}"
        self._log_reward_validation(rollout_transitions, rollout_prefix)
        self._log_return_ranking(self.trajectories, rollout_prefix)
        if self.debug_dataset:
            debug_prefix = f"reward_val/debug_dataset/{stage}"
            self._log_reward_validation(self.debug_dataset, debug_prefix)
            self._log_return_ranking(self._debug_trajectories, debug_prefix)

    def _log_reward_validation(self, transitions, log_class: str) -> None:
        true_rewards, pred_rewards, pred_std, status = self._run_reward_inference_with_std(transitions)
        running_only = status[:, self.STATUS_RUNNING] == 1
        self._record_reward_correlation(f"{log_class}/pred_true", true_rewards, pred_rewards, running_only)
        arrived_mask = status[:, self.STATUS_ARRIVED] == 1
        collided_mask = status[:, self.STATUS_COLLIDED] == 1
        offroad_mask = status[:, self.STATUS_OFFROAD] == 1
        timeout_mask = status[:, self.STATUS_TIMEOUT] == 1
        running_mask = status[:, self.STATUS_RUNNING] == 1

        self.logger.record(f"{log_class}/reward_mean", float(np.mean(pred_rewards)))
        self.logger.record(f"{log_class}/reward_std", float(np.std(pred_rewards)))
        self.logger.record(f"{log_class}/reward_min", float(np.min(pred_rewards)))
        self.logger.record(f"{log_class}/reward_max", float(np.max(pred_rewards)))
        running_mean = self._record_masked_mean(f"{log_class}/reward_running", pred_rewards, running_mask)
        arrived_mean = self._record_masked_mean(f"{log_class}/reward_arrived", pred_rewards, arrived_mask)
        collided_mean = self._record_masked_mean(f"{log_class}/reward_collided", pred_rewards, collided_mask)
        self._record_masked_mean(f"{log_class}/reward_offroad", pred_rewards, offroad_mask)
        self._record_masked_mean(f"{log_class}/reward_timeout", pred_rewards, timeout_mask)
        if arrived_mean is not None and collided_mean is not None:
            self.logger.record(f"{log_class}/gap_arrived_collided", arrived_mean - collided_mean)
        if arrived_mean is not None and running_mean is not None:
            self.logger.record(f"{log_class}/gap_arrived_running", arrived_mean - running_mean)
        self.logger.record(f"{log_class}/ensemble_std", float(np.mean(pred_std)))
        self._record_masked_mean(f"{log_class}/ensemble_std_running", pred_std, running_mask)

    def _record_reward_correlation(
        self,
        key_prefix: str,
        true_rewards: np.ndarray,
        pred_rewards: np.ndarray,
        running_mask: np.ndarray,
    ) -> None:
        """Log per-step pred vs true reward correlation.

        This is the reward model's primary health metric: with soft preference
        labels the Bradley-Terry loss sits at its ln(2) cross-entropy floor
        even when learning succeeds, so correlation (and preference accuracy)
        are what to watch instead.
        """
        for suffix, mask in (("all", np.ones(len(true_rewards), dtype=bool)), ("running", running_mask)):
            t, p = true_rewards[mask], pred_rewards[mask]
            if len(t) < 2 or np.std(t) < 1e-12 or np.std(p) < 1e-12:
                continue
            pearson, _ = pearsonr(t, p)
            spearman, _ = spearmanr(t, p)
            if np.isfinite(pearson):
                self.logger.record(f"{key_prefix}/pearson_{suffix}", float(pearson))
            if np.isfinite(spearman):
                self.logger.record(f"{key_prefix}/spearman_{suffix}", float(spearman))

    def _record_masked_mean(self, key: str, values: np.ndarray, mask: np.ndarray):
        if np.any(mask):
            mean = float(np.mean(values[mask]))
            self.logger.record(key, mean)
            return mean
        return None

    def _run_reward_inference_with_std(self, transitions):
        self.reward_model.eval()
        with th.no_grad():
            true_rewards = np.array([t.true_reward for t in transitions], dtype=np.float32)
            obs = np.array([t.observation for t in transitions], dtype=np.float32)
            acts = np.array([t.action for t in transitions], dtype=np.float32)
            status = np.array([t.next_status for t in transitions], dtype=np.float32)
            done = np.array([float(t.done) for t in transitions], dtype=np.float32)
            pred_rewards = self.reward_model.predict(obs, acts, status, done)
            pred_std = self.reward_model.predict_all(obs, acts, status, done).std(axis=1)
            if not np.isfinite(pred_rewards).all() or not np.isfinite(pred_std).all():
                raise FloatingPointError("Non-finite reward-model validation predictions.")
        self.reward_model.train()
        return true_rewards, pred_rewards, pred_std, status

    @staticmethod
    def _split_into_trajectories(transitions):
        trajectories, current = [], Trajectory()
        for transition in transitions:
            current.add_transition(transition)
            if transition.done:
                trajectories.append(current)
                current = Trajectory()
        if len(current) > 0:
            trajectories.append(current)
        return trajectories

    def _log_return_ranking(self, trajectories, log_class: str) -> None:
        if len(trajectories) < 2:
            return
        # One batched inference over all transitions, split back per trajectory.
        all_transitions = [t for traj in trajectories for t in traj]
        _, pred_rewards, _, _ = self._run_reward_inference_with_std(all_transitions)
        lengths = [len(traj) for traj in trajectories]
        boundaries = np.cumsum(lengths)[:-1]
        pred_returns = [float(chunk.sum()) for chunk in np.split(pred_rewards, boundaries)]
        true_returns = [float(sum(t.true_reward for t in traj)) for traj in trajectories]
        rho, _ = spearmanr(true_returns, pred_returns)
        is_defined = float(np.isfinite(rho))
        self.logger.record(f"{log_class}/spearman_returns_defined", is_defined)
        if is_defined:
            self.logger.record(f"{log_class}/spearman_returns", float(rho))
