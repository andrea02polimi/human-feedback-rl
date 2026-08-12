"""Reward-loss, validation, ranking, and replay-buffer diagnostics."""

import numpy as np
import torch as th
import wandb
from scipy.stats import pearsonr, spearmanr

from human_feedback_rl.common.batching import fragment_sum_rewards
from human_feedback_rl.common.types import Trajectory


class RewardDiagnosticsMixin:
    """Non-training reward diagnostics used by ``HybridAlgorithm``."""

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

            if self.loss_type == "demo_1":
                loss = expert_term + model_mean
                self.logger.record("reward/loss", float(loss), exclude="stdout")
                self.logger.record("reward/demo_1_margin", float(margin), exclude="stdout")
                self.logger.record("reward/demo_1_scale_std", float(all_returns.std(unbiased=False)), exclude="stdout")
                self.logger.record("reward/demo_1_scale_abs", float(abs_mean), exclude="stdout")
                self.reward_model.train()
                return

            # demo_2: expert+model partition estimate.
            partition = th.logsumexp(all_returns, dim=0) - np.log(len(all_returns))
            weights = th.softmax(all_returns, dim=0)
            expert_mass = weights[:len(expert_returns)].sum()
            model_mass = weights[len(expert_returns):].sum()
            top1_weight, top1_idx = weights.max(dim=0)
            ess = 1.0 / weights.pow(2).sum()
            self.logger.record("reward/loss", float(expert_term + partition), exclude="stdout")
            self.logger.record("reward/demo_2_partition_all", float(partition), exclude="stdout")
            self.logger.record("reward/demo_2_expert_softmax_mass", float(expert_mass), exclude="stdout")
            self.logger.record("reward/demo_2_model_softmax_mass", float(model_mass), exclude="stdout")
            self.logger.record("reward/demo_2_top1_softmax_weight", float(top1_weight), exclude="stdout")
            self.logger.record("reward/demo_2_top1_is_expert", float((top1_idx < len(expert_returns)).item()), exclude="stdout")
            self.logger.record("reward/demo_2_effective_sample_size", float(ess), exclude="stdout")
            self._log_ess_fraction("reward/demo_2", ess, len(weights))
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

    def _log_return_scatter(self, trajectories, log_class: str, iteration: int) -> None:
        """Log a W&B image scatter of per-trajectory returns: x = predicted
        (current reward model), y = true.

        Logged as ``wandb.Image`` (not ``wandb.plot.scatter``) so every step is
        retained in the media panel: the slider scrubs the historical scatters
        instead of overwriting them. ``commit=False`` attaches the image to the
        iteration's pending log so it shares the step of the rest of the metrics.
        The y-limits are derived from the (fixed) true returns so every frame
        shares a stable vertical reference; x is the predicted return after a
        global affine alignment to the true-reward scale (running steps).
        """
        if wandb.run is None or len(trajectories) < 2:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        class_names = {
            self.STATUS_ARRIVED: "arrived",
            self.STATUS_COLLIDED: "collided",
            self.STATUS_OFFROAD: "offroad",
            self.STATUS_TIMEOUT: "timeout",
        }

        colors = {
            "arrived": "tab:green",
            "collided": "tab:red",
            "offroad": "tab:orange",
            "timeout": "tab:blue",
            "other": "tab:gray",
        }

        true_returns, outcomes = [], []
        pred_step_lists, true_step_lists, running_lists, traj_lengths = [], [], [], []

        for traj in trajectories:
            true_rewards, pred_rewards, _, status = self._run_reward_inference_with_std(traj)
            true_returns.append(float(true_rewards.sum()))
            pred_step_lists.append(np.asarray(pred_rewards, dtype=np.float64))
            true_step_lists.append(np.asarray(true_rewards, dtype=np.float64))
            running_lists.append(status[:, self.STATUS_RUNNING] == 1)
            traj_lengths.append(len(pred_rewards))

            final_status = np.asarray(traj[-1].next_status)
            outcome = (
                class_names.get(int(np.argmax(final_status)), "other")
                if np.isclose(final_status.sum(), 1.0)
                else "other"
            )
            outcomes.append(outcome)

        # A single global affine maps step-level predictions onto the true-reward
        # scale, fitted on RUNNING steps only (terminal rewards stay the signal we
        # want to reconstruct). One shift/scale shared by every trajectory, so the
        # cross-trajectory ranking is untouched and only the x-axis units change.
        # Nessuna temperatura: non e' un attributo dell'algoritmo (vive nel
        # gatherer) e l'affine la cancellerebbe comunque, essendo fittata
        # sulle stesse predizioni a cui verrebbe applicata.
        pred_steps = np.concatenate(pred_step_lists)
        true_steps = np.concatenate(true_step_lists)
        running = np.concatenate(running_lists)
        ref = running if running.any() else np.ones(len(pred_steps), dtype=bool)

        pred_mean, true_mean = pred_steps[ref].mean(), true_steps[ref].mean()
        pred_std, true_std = pred_steps[ref].std(), true_steps[ref].std()
        scale = (true_std / pred_std) if pred_std > 1e-8 else 1.0
        shift = true_mean - scale * pred_mean

        pred_returns = np.asarray([
            scale * p.sum() + shift * n
            for p, n in zip(pred_step_lists, traj_lengths)
        ])

        fig, ax = plt.subplots(figsize=(5, 5))

        true_returns = np.asarray(true_returns)
        pred_returns = np.asarray(pred_returns)
        outcomes = np.asarray(outcomes)

        for outcome, color in colors.items():
            mask = outcomes == outcome
            if mask.any():
                ax.scatter(
                    pred_returns[mask],
                    true_returns[mask],
                    color=color,
                    label=outcome,
                    alpha=0.7,
                )

        ax.legend(title="Final outcome")
        ax.set_xlabel("predicted return (aligned to true, running-step affine)")
        ax.set_ylabel("true return")
        ax.set_title(f"{log_class} iter {iteration}")
        y_min, y_max = min(true_returns), max(true_returns)
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        try:
            wandb.log({
                "iterations": iteration,
                f"{log_class}/return_scatter": wandb.Image(fig),
            }, commit=True)
        finally:
            plt.close(fig)

    def _log_outcome_returns(self) -> None:
        """Log the discounted return under the *current* reward, by terminal outcome.

        For each terminal status records mean raw, normalized and discounted
        return plus per-step/terminal reward and length. ``disc_return`` is the
        discounted sum of the current-model normalized reward: it equals what SAC
        optimizes only when ``relabel_rewards=True`` (otherwise the critic uses
        stored rewards from older models) and it excludes SAC's entropy term — so
        it is a faithful *proxy*, not SAC's exact objective. Still, if offroad has
        a higher disc_return than arrived, the reward — not SAC — is the problem.

        CAVEAT — for ``timeout`` ``disc_return`` is only a *partial* proxy: SAC
        treats timeouts as truncations and bootstraps the value beyond the
        episode, while this sum stops at the last step. The comparison is sound
        for arrived/offroad/collided (true terminations, no bootstrap).
        """
        if not self.trajectories:
            return
        gamma = float(getattr(self.agent, "gamma", 1.0))
        status_names = {
            self.STATUS_ARRIVED: "arrived",
            self.STATUS_COLLIDED: "collided",
            self.STATUS_OFFROAD: "offroad",
            self.STATUS_TIMEOUT: "timeout",
            5: "teleported",
            6: "removed_unknown",
        }
        buckets = {name: [] for name in status_names.values()}
        self.reward_model.eval()
        with th.no_grad():
            for traj in self.trajectories:
                if len(traj) == 0:
                    continue
                # Only classify genuine terminal transitions with a valid one-hot;
                # an all-zero status would otherwise be misread as "arrived".
                last_status = np.asarray(traj[-1].next_status, dtype=np.float64)
                if not traj[-1].done or not np.isclose(last_status.sum(), 1.0):
                    continue
                name = status_names.get(int(np.argmax(last_status)))
                if name is None:
                    continue
                obs = np.array([t.observation for t in traj], dtype=np.float32)
                acts = np.array([t.action for t in traj], dtype=np.float32)
                status = np.array([t.next_status for t in traj], dtype=np.float32)
                done = np.array([float(t.done) for t in traj], dtype=np.float32)
                raw = self.reward_model.predict_unnormalized(obs, acts, status, done)
                norm = self.reward_model.predict(obs, acts, status, done)
                discounts = gamma ** np.arange(len(norm), dtype=np.float64)
                buckets[name].append((
                    float(raw.sum()), float(norm.sum()), float(np.sum(norm * discounts)),
                    float(norm.mean()), float(norm[-1]), float(len(traj)),
                ))
        self.reward_model.train()

        fields = ("raw_return", "norm_return", "disc_return",
                  "mean_step_reward", "terminal_reward", "length")
        for name, rows in buckets.items():
            # Always log count (even 0) so "no episodes of this type" is
            # distinguishable from "metric not run"; skip only the means.
            self.logger.record(f"reward/outcome/{name}/count", len(rows), exclude="stdout")
            if not rows:
                continue
            values = np.asarray(rows, dtype=np.float64)
            for i, field in enumerate(fields):
                self.logger.record(
                    f"reward/outcome/{name}/{field}", float(values[:, i].mean()), exclude="stdout"
                )
