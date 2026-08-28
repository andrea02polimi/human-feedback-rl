"""Fitting the reward model to the feedback collected so far."""

import time
import numpy as np
import torch as th
from human_feedback_rl.common.batching import fragment_avg_rewards
from human_feedback_rl.common.datasets import PreferenceBatch
from human_feedback_rl.common.preference_losses import (
    bradley_terry_probs,
    evaluate_preference_batch,
    preference_labels_tensor,
    preference_nll,
)


class RewardModelTrainingMixin:
    """The reward-model training loop, in both demo modes."""

    def _train_reward_model(self) -> None:
        if not self.trajectories:
            return
        if self.demo_mode == "preferences":
            self._train_reward_model_pure_preferences()
        else:
            self._train_reward_model_gcl()

    def _training_view(self, dataset):
        """The dataset this iteration draws its minibatches from.

        The bootstrap is there to decorrelate ensemble members, so a single member
        skips it. Resampling n items out of n keeps only 63.2% distinct, which would
        throw away a third of the comparisons for no benefit.
        """
        if self.bootstrap_comparisons is None:
            resample = len(self.reward_model.members) > 1
        else:
            resample = bool(self.bootstrap_comparisons)
        return dataset.bootstrap() if resample else dataset

    def _train_reward_model_gcl(self) -> None:
        """BT + GCL on the shared net with norm-balanced gradient fusion.

        Degenerate cases stay well-defined: with an empty preference dataset
        only the demo loss trains (demo-only arm); with ``demo_weight == 0``
        only the preference loss trains (pref-only arm).
        """
        has_prefs = len(self.dataset_train) > 0
        demo_weight = self.demo_weight
        if not has_prefs and demo_weight == 0.0:
            return

        self.logger.record("reward/demo_weight", demo_weight, exclude="stdout")
        # Before any step: the weight has to describe this theta.
        self._estimate_alpha()
        t0 = time.perf_counter()

        def member_step(member, optimizer):
            member.train()
            boot_dataset = self._training_view(self.dataset_train) if has_prefs else None
            # One alpha per member for the whole iteration, estimated above
            # at these same parameters.
            alpha = self._alpha_weight(member)
            stats = []
            for _ in range(self.gradient_steps_rew):
                pref_loss = (
                    self._preference_loss(member, boot_dataset.sample(self.batch_size_pref))
                    if boot_dataset is not None
                    else None
                )
                demo_loss = self._reward_loss(member) if demo_weight > 0.0 else None
                stats.append(
                    self._reward_step(member, optimizer, pref_loss, demo_loss, alpha=alpha)
                )
            return stats

        all_stats = [s for stats in self.train_reward_members(member_step) for s in stats]
        t_train = time.perf_counter() - t0

        self._log_reward_loss_diagnostics()
        self._log_preference_diagnostics()
        self._log_hybrid_step_stats(all_stats)
        self.logger.record("reward/weight_norm", self._param_norm(self.reward_model), exclude="stdout")
        self.logger.record("time/train_reward_model", t_train)

    def _train_reward_model_pure_preferences(self) -> None:
        """Single BT objective on mixed oracle + expert-vs-agent batches."""
        has_oracle = len(self.dataset_train) > 0
        has_demo = len(self.dataset_demo_prefs_train) > 0
        if not has_oracle and not has_demo:
            return

        t0 = time.perf_counter()
        n_demo = round(self.batch_size_pref * self.demo_pref_batch_fraction)
        if not has_demo:
            n_demo = 0
        if not has_oracle:
            n_demo = self.batch_size_pref
        n_oracle = self.batch_size_pref - n_demo

        def member_step(member, optimizer):
            member.train()
            boot_oracle = self._training_view(self.dataset_train) if n_oracle else None
            boot_demo = self._training_view(self.dataset_demo_prefs_train) if n_demo else None
            losses = []
            for _ in range(self.gradient_steps_rew):
                parts = []
                if boot_oracle is not None:
                    parts.append(boot_oracle.sample(n_oracle))
                if boot_demo is not None:
                    parts.append(boot_demo.sample(n_demo))
                batch = PreferenceBatch(
                    [pair for part in parts for pair in part.fragment_pairs],
                    [pref for part in parts for pref in part.preferences],
                )
                loss = self._preference_loss(member, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            return losses

        all_losses = [l for losses in self.train_reward_members(member_step) for l in losses]
        t_train = time.perf_counter() - t0

        self._log_preference_diagnostics()
        self.logger.record("reward/hybrid_pref_loss", float(np.mean(all_losses)), exclude="stdout")
        self.logger.record("reward/weight_norm", self._param_norm(self.reward_model), exclude="stdout")
        self.logger.record("time/train_reward_model", t_train)

    def _preference_loss(self, member, batch) -> th.Tensor:
        r1 = fragment_avg_rewards(member, [p.frag1 for p in batch.fragment_pairs])
        r2 = fragment_avg_rewards(member, [p.frag2 for p in batch.fragment_pairs])
        labels = self._smoothed_labels(preference_labels_tensor(batch.preferences))
        return preference_nll(bradley_terry_probs(r1, r2), labels)

    def _smoothed_labels(self, labels: th.Tensor) -> th.Tensor:
        """Move sampled binary labels label_smoothing of the way towards 1/2.

        Against a target in {0, 1} the cross-entropy minimum sits at infinity, so few
        comparisons get memorised instead of learned. Soft labels already have a finite
        optimum and are left alone.
        """
        if self.label_smoothing <= 0.0 or self.labels_type != "binary_bernoulli":
            return labels
        eps = self.label_smoothing
        return (1.0 - eps) * labels + eps / 2.0

    def _log_preference_diagnostics(self) -> None:
        """Fit diagnostics on the training comparisons.

        There is no validation set: feedback is the scarce resource here, and
        holding a share of it back took it away from training while no decision
        really depended on that measurement. The ``*_val`` keys disappear with
        it.
        """
        if not len(self.dataset_train):
            return
        train_loss, train_acc = evaluate_preference_batch(
            self.reward_model, self.dataset_train.get_all()
        )
        self.logger.record("reward/loss_pref_train", train_loss, exclude="stdout")
        self.logger.record("reward/acc_pref_train", train_acc, exclude="stdout")
        if self.demo_mode == "preferences" and len(self.dataset_demo_prefs_train):
            _, demo_acc = evaluate_preference_batch(
                self.reward_model, self.dataset_demo_prefs_train.get_all()
            )
            self.logger.record("reward/acc_demo_pref_train", demo_acc, exclude="stdout")
