"""Reward-loss, validation, ranking, and replay-buffer diagnostics."""

import numpy as np
import torch as th
import torch.nn.functional as F
from scipy.stats import spearmanr

from human_feedback_rl.common.types import Trajectory


class RewardDiagnosticsMixin:
    """Non-training reward diagnostics used by ``DemoAlgorithm``."""

    def _diagnostic_returns(self, member):
        expert_trajs = self._even_subset(self.expert_trajectories, self.batch_size_expert)
        model_trajs = self._even_subset(self.trajectories, self.batch_size_model)
        expert_returns = th.stack([self._traj_sum_reward(member, traj) for traj in expert_trajs])
        model_returns = th.stack([self._traj_sum_reward(member, traj) for traj in model_trajs])
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

    def _log_reward_loss_diagnostics(self) -> None:
        if not self.trajectories:
            return

        self.reward_model.eval()
        with th.no_grad():
            expert_returns, model_returns, expert_trajs, model_trajs = self._diagnostic_returns(
                self.reward_model
            )
            expert_term = -expert_returns.mean()
            model_mean = model_returns.mean()
            all_returns = th.cat([expert_returns, model_returns], dim=0)
            if not th.isfinite(all_returns).all():
                raise FloatingPointError("Non-finite trajectory returns in reward diagnostics.")
            margin = expert_returns.mean() - model_mean
            abs_mean = all_returns.abs().mean()

            self.logger.record("reward/expert_return_mean", float(expert_returns.mean()), exclude="stdout")
            self.logger.record("reward/model_return_mean", float(model_mean), exclude="stdout")
            self.logger.record("reward/expert_model_margin", float(margin), exclude="stdout")
            self.logger.record("reward/return_std", float(all_returns.std(unbiased=False)), exclude="stdout")
            self.logger.record("reward/return_abs_mean", float(abs_mean), exclude="stdout")
            self.logger.record("reward/return_min", float(all_returns.min()), exclude="stdout")
            self.logger.record("reward/return_max", float(all_returns.max()), exclude="stdout")

            if self.loss_type in ("demo", "demo_loss"):
                loss = expert_term + model_mean
                self.logger.record("reward/loss", float(loss), exclude="stdout")
                self.logger.record("reward/demo_margin", float(margin), exclude="stdout")
                self.logger.record("reward/demo_scale_std", float(all_returns.std(unbiased=False)), exclude="stdout")
                self.logger.record("reward/demo_scale_abs", float(abs_mean), exclude="stdout")
                self.reward_model.train()
                return

            if self.loss_type == "demo_corrected":
                demo_margin = self._demo_corrected_margins(
                    expert_returns, model_returns, expert_trajs, model_trajs
                )
                loss = F.softplus(-demo_margin / self.temperature).mean()
                self.logger.record("reward/loss", float(loss), exclude="stdout")
                self.logger.record("reward/demo_corrected_margin", float(demo_margin.mean()), exclude="stdout")
                self.logger.record("reward/demo_corrected_scale_std", float(all_returns.std(unbiased=False)), exclude="stdout")
                self.reward_model.train()
                return

            if self.loss_type == "maxent_2":
                partition = th.logsumexp(all_returns, dim=0) - np.log(len(all_returns))
                weights = th.softmax(all_returns, dim=0)
                expert_mass = weights[:len(expert_returns)].sum()
                model_mass = weights[len(expert_returns):].sum()
                top1_weight, top1_idx = weights.max(dim=0)
                ess = 1.0 / weights.pow(2).sum()
                self.logger.record("reward/loss", float(expert_term + partition), exclude="stdout")
                self.logger.record("reward/maxent2_partition_all", float(partition), exclude="stdout")
                self.logger.record("reward/maxent2_expert_softmax_mass", float(expert_mass), exclude="stdout")
                self.logger.record("reward/maxent2_model_softmax_mass", float(model_mass), exclude="stdout")
                self.logger.record("reward/maxent2_top1_softmax_weight", float(top1_weight), exclude="stdout")
                self.logger.record("reward/maxent2_top1_is_expert", float((top1_idx < len(expert_returns)).item()), exclude="stdout")
                self.logger.record("reward/maxent2_effective_sample_size", float(ess), exclude="stdout")
                self._log_ess_fraction("reward/maxent2", ess, len(weights))
                self.reward_model.train()
                return

            log_q = None
            partition_logits = model_returns
            log_prefix = "reward/maxent"
            loss_expert_term = expert_term
            if self.loss_type == "maxent_corrected":
                # Mirror the fragment-level partition used by the loss so the
                # logged partition/ESS/loss match what training actually optimises.
                frag_expert_returns = self._fragment_returns(self.reward_model, expert_trajs)
                frag_model_returns = self._fragment_returns(self.reward_model, model_trajs)
                log_q = self._fragment_log_probs(model_trajs)
                partition_logits = frag_model_returns / self.temperature - log_q
                log_prefix = "reward/maxent_corrected"
                loss_expert_term = -frag_expert_returns.mean() / self.temperature

            partition = th.logsumexp(partition_logits, dim=0) - np.log(len(partition_logits))
            weights = th.softmax(partition_logits, dim=0)
            top_values = th.topk(weights, k=min(5, len(weights))).values
            ess = 1.0 / weights.pow(2).sum()
            loss = loss_expert_term + partition

            self.logger.record("reward/loss", float(loss), exclude="stdout")
            self.logger.record(f"{log_prefix}_partition_model", float(partition), exclude="stdout")
            self.logger.record(f"{log_prefix}_top1_softmax_weight", float(weights.max()), exclude="stdout")
            self.logger.record(f"{log_prefix}_top5_softmax_mass", float(top_values.sum()), exclude="stdout")
            self.logger.record(f"{log_prefix}_effective_sample_size", float(ess), exclude="stdout")
            self._log_ess_fraction(log_prefix, ess, len(weights))
            if log_q is not None:
                self.logger.record(f"{log_prefix}_log_q_mean", float(log_q.mean()), exclude="stdout")
        self.reward_model.train()

    def _log_ess_fraction(self, log_prefix: str, ess: th.Tensor, n_items: int) -> None:
        ess_fraction = float(ess) / n_items
        self.logger.record(
            f"{log_prefix}_effective_sample_fraction", ess_fraction, exclude="stdout"
        )
        if ess_fraction < 0.1:
            self.logger.warn(
                f"Low effective sample fraction for {self.loss_type}: {ess_fraction:.3f}"
            )

    def _log_replay_reward_staleness(self, batch_size: int = 2048) -> None:
        replay_buffer = getattr(self.agent, "replay_buffer", None)
        if replay_buffer is None or not hasattr(replay_buffer, "sample_reward_staleness"):
            return
        batch = replay_buffer.sample_reward_staleness(batch_size, self._debug_rng)
        if batch is None:
            return

        stored_rewards, current_rewards = batch
        delta = current_rewards - stored_rewards
        abs_delta = np.abs(delta)
        stored_std = float(np.std(stored_rewards))
        current_std = float(np.std(current_rewards))
        denom = current_std if current_std > 1e-8 else 1.0

        self.logger.record("replay_relabel_debug/sample_size", len(stored_rewards), exclude="stdout")
        self.logger.record("replay_relabel_debug/stored_reward_mean", float(np.mean(stored_rewards)), exclude="stdout")
        self.logger.record("replay_relabel_debug/current_reward_mean", float(np.mean(current_rewards)), exclude="stdout")
        self.logger.record("replay_relabel_debug/stored_reward_std", stored_std, exclude="stdout")
        self.logger.record("replay_relabel_debug/current_reward_std", current_std, exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_mean", float(np.mean(delta)), exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_std", float(np.std(delta)), exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_abs_mean", float(np.mean(abs_delta)), exclude="stdout")
        self.logger.record("replay_relabel_debug/delta_abs_p95", float(np.percentile(abs_delta, 95)), exclude="stdout")
        self.logger.record("replay_relabel_debug/staleness_ratio", float(np.mean(abs_delta) / denom), exclude="stdout")
        self.logger.record("replay_relabel_debug/sign_flip_frac", float(np.mean(np.sign(stored_rewards) != np.sign(current_rewards))), exclude="stdout")
        relabel_enabled = float(getattr(replay_buffer, "relabel_rewards", False))
        self.logger.record("replay_relabel_debug/relabel_enabled", relabel_enabled, exclude="stdout")
        self.logger.record("replay_relabel_debug/critic_uses_current_reward", relabel_enabled, exclude="stdout")
        if stored_std > 1e-8 and current_std > 1e-8:
            corr = np.corrcoef(stored_rewards, current_rewards)[0, 1]
            self.logger.record("replay_relabel_debug/stored_current_corr", float(corr), exclude="stdout")

    def _log_validation_snapshot(self, rollout_transitions, stage: str) -> None:
        rollout_prefix = f"reward_val/current_rollout/{stage}"
        self._log_reward_validation(rollout_transitions, rollout_prefix)
        self._log_return_ranking(self.trajectories, rollout_prefix)
        if self.debug_dataset:
            debug_prefix = f"reward_val/debug_dataset/{stage}"
            self._log_reward_validation(self.debug_dataset, debug_prefix)
            self._log_return_ranking(self._debug_trajectories, debug_prefix)

    def _log_reward_validation(self, transitions, log_class: str) -> None:
        _, pred_rewards, pred_std, status = self._run_reward_inference(transitions)
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

    def _record_masked_mean(self, key: str, values: np.ndarray, mask: np.ndarray):
        if np.any(mask):
            mean = float(np.mean(values[mask]))
            self.logger.record(key, mean)
            return mean
        return None

    def _run_reward_inference(self, transitions):
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
        true_returns, pred_returns = [], []
        for traj in trajectories:
            _, pred_rewards, _, _ = self._run_reward_inference(traj)
            true_returns.append(float(sum(t.true_reward for t in traj)))
            pred_returns.append(float(pred_rewards.sum()))
        rho, _ = spearmanr(true_returns, pred_returns)
        is_defined = float(np.isfinite(rho))
        self.logger.record(f"{log_class}/spearman_returns_defined", is_defined)
        if is_defined:
            self.logger.record(f"{log_class}/spearman_returns", float(rho))
