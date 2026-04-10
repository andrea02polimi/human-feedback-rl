import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from human_feedback_rl.common import (
    PreferenceDataset,
    PreferenceModelFromReward,
    UnifiedLogger,
)


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _preference_to_target(preference, device) -> torch.Tensor:
    """Convert a Preference label to a class-index tensor."""
    idx = preference.label.index(max(preference.label))
    return torch.tensor([idx], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Reward Trainer (Christiano et al. 2017)
# ---------------------------------------------------------------------------

class RewardTrainerChristiano:
    """
    Trainer for an ensemble reward model using preference comparisons.

    Each ensemble member is trained on an independent bootstrap sample of the
    dataset (sampling with replacement), maximising ensemble diversity and
    producing calibrated uncertainty estimates for active learning.

    Mirrors ``imitation.algorithms.preference_comparisons.BasicRewardTrainer`` /
    ``EnsembleTrainer`` in API: ``train(dataset, epoch_multiplier)`` scales the
    number of training epochs relative to ``self.epochs`` (base count set at
    construction time).  The ``num_epochs`` keyword is kept for backward
    compatibility with ``ChristianoPPOAlgorithm`` and ``ChristianoSACAlgorithm``.
    """

    def __init__(
        self,
        preference_model: PreferenceModelFromReward,
        epochs: int = 1,
        batch_size: int = 32,
        logger: UnifiedLogger = None,
    ):
        """
        Args:
            preference_model: preference model wrapping the ensemble reward net.
            epochs: base number of training epochs per call to ``train()``.
                Scaled by ``epoch_multiplier`` in ``train()`` — mirrors
                ``BasicRewardTrainer(epochs=...)`` in imitation.
            batch_size: mini-batch size for each ensemble member.
            logger: logger for loss / epoch metrics.
        """
        self.preference_model = preference_model
        self.reward_model = preference_model.reward_model

        self.epochs = epochs
        self.batch_size = batch_size
        self.logger = logger
        self.global_epochs = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def train(
        self,
        dataset: PreferenceDataset,
        epoch_multiplier: float = 1.0,
        num_epochs: Optional[int] = None,
    ) -> dict:
        """Train on a preference dataset.

        Mirrors ``imitation.algorithms.preference_comparisons.RewardTrainer.train(
        dataset, epoch_multiplier)``.

        Args:
            dataset: preference pairs to train on.
            epoch_multiplier: scale factor applied to ``self.epochs``.
                On iteration 0, ``PreferenceComparisons`` passes
                ``initial_epoch_multiplier`` so the reward model is calibrated
                before agent training begins.  Ignored when ``num_epochs`` is set.
            num_epochs: explicit epoch count (backward-compat with SAC/PPO
                algorithms).  If given, ``epoch_multiplier`` is ignored.

        Returns:
            ``{"loss": float, "accuracy": float}`` — averages over the last epoch.
        """
        if num_epochs is None:
            num_epochs = max(1, round(self.epochs * epoch_multiplier))

        if len(dataset) == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        last_metrics: dict = {"loss": 0.0, "accuracy": 0.0}

        for epoch in range(num_epochs):
            # Each ensemble member trains on an independent bootstrap sample
            # (bagging), mirroring imitation's EnsembleTrainer._train().
            n = len(dataset)
            bootstrap_indices_per_ensemble = [
                np.random.choice(n, size=n, replace=True).tolist()
                for _ in range(self.reward_model.n_ensembles)
            ]

            last_metrics = self._train_one_epoch(dataset, bootstrap_indices_per_ensemble)
            self.global_epochs += 1

            self.logger.record("loss",         last_metrics["loss"])
            self.logger.record("accuracy",     last_metrics["accuracy"])
            self.logger.record("global_epochs", self.global_epochs)
            self.logger.dump()

        return last_metrics


    def evaluate(self, dataset: PreferenceDataset) -> float:
        """Evaluate reward model accuracy/loss on a preference dataset."""
        return RewardModelEvaluator(self.preference_model).evaluate(dataset)

    # -----------------------------------------------------------------------
    # Internal logic
    # -----------------------------------------------------------------------

    def _train_one_epoch(
        self,
        dataset: PreferenceDataset,
        bootstrap_indices_per_ensemble: list,
    ) -> dict:
        """Run one full epoch over the bootstrapped dataset.

        Returns ``{"loss": float, "accuracy": float}`` averaged over all batches.
        """
        epoch_loss = 0.0
        epoch_acc  = 0.0
        n_batches  = 0
        n_total    = len(bootstrap_indices_per_ensemble[0])

        for batch_start in range(0, n_total, self.batch_size):
            batch_indices_per_ensemble = [
                indices[batch_start : batch_start + self.batch_size]
                for indices in bootstrap_indices_per_ensemble
            ]
            batch_size_actual = len(batch_indices_per_ensemble[0])

            metrics = self._train_on_batch(dataset, batch_indices_per_ensemble)

            # Scale loss for incomplete batches so the gradient magnitude is
            # proportional to the actual batch fraction — mirrors imitation's
            # ``loss *= len(fragment_pairs) / self.batch_size`` rescaling.
            scale = batch_size_actual / self.batch_size
            epoch_loss += metrics["loss"] * scale
            epoch_acc  += metrics["accuracy"]
            n_batches  += 1

        return {
            "loss":     epoch_loss / max(n_batches, 1),
            "accuracy": epoch_acc  / max(n_batches, 1),
        }

    def _train_on_batch(
        self,
        dataset: PreferenceDataset,
        batch_indices_per_ensemble: list,
    ) -> dict:
        """Train all ensemble members on one mini-batch.

        Returns ``{"loss": float, "accuracy": float}`` for this batch.
        The accuracy is computed from ensemble-mean predictions (no gradient),
        mirroring imitation's ``CrossEntropyRewardLoss`` accuracy metric.
        """
        # ------------------------------------------------------------------
        # Accuracy: ensemble-mean prediction, computed before the update so
        # it reflects the model BEFORE this batch's gradient step.
        # Uses the pair indices from ensemble member 0's bootstrap (representative).
        # ------------------------------------------------------------------
        pair_indices = batch_indices_per_ensemble[0]
        correct = 0
        with torch.no_grad():
            for idx in pair_indices:
                pair = dataset.pairs[idx]
                pref = dataset.preferences[idx]
                # preference_probs averages across ensemble members (Bradley-Terry)
                prob = self.preference_model.preference_probs(pair.seg1, pair.seg2)
                pred_pref_1  = prob.label[0] > 0.5
                label_pref_1 = pref.label[0]  > 0.5
                correct += int(pred_pref_1 == label_pref_1)
        accuracy = correct / len(pair_indices) if pair_indices else 0.0

        # ------------------------------------------------------------------
        # Loss + backward: one optimizer step per ensemble member.
        # ------------------------------------------------------------------
        total_batch_loss = 0.0
        for ensemble_idx, batch_indices in enumerate(batch_indices_per_ensemble):
            opt = self.reward_model.optimizers[ensemble_idx]
            opt.zero_grad()

            losses = [
                self._compute_pair_loss(
                    dataset.pairs[idx],
                    _preference_to_target(dataset.preferences[idx], self.reward_model.device),
                    ensemble_idx,
                )
                for idx in batch_indices
            ]

            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            opt.step()

            total_batch_loss += batch_loss.item()

        return {
            "loss":     total_batch_loss / len(batch_indices_per_ensemble),
            "accuracy": accuracy,
        }

    # -----------------------------------------------------------------------
    # Core computations
    # -----------------------------------------------------------------------

    def _compute_pair_loss(self, pair, target, ensemble_idx: int) -> torch.Tensor:
        """Cross-entropy loss for a single ensemble member."""
        r1, r2 = self.preference_model.preference_logits_for_net(
            pair.seg1, pair.seg2, ensemble_idx
        )
        return F.cross_entropy(torch.stack([r1, r2]).unsqueeze(0), target)


# ---------------------------------------------------------------------------
# Evaluation Module
# ---------------------------------------------------------------------------

class RewardModelEvaluator:
    """Evaluate a reward model on a preference dataset (accuracy + loss)."""

    def __init__(self, preference_model: PreferenceModelFromReward):
        self.preference_model = preference_model
        self.reward_model = preference_model.reward_model

    def evaluate(self, dataset: PreferenceDataset) -> float:
        if len(dataset) == 0:
            return 0.0

        total_loss = 0.0

        with torch.no_grad():
            for pair, pref in dataset:
                target = _preference_to_target(pref, self.reward_model.device)
                logits = self._ensemble_logits(pair)
                total_loss += F.cross_entropy(logits, target).item()

        return total_loss / len(dataset)

    def _ensemble_logits(self, pair) -> torch.Tensor:
        """Average logits across ensemble members. Shape: (1, 2)."""
        logits_list = [
            torch.stack(
                self.preference_model.preference_logits_for_net(pair.seg1, pair.seg2, k)
            )
            for k in range(self.reward_model.n_ensembles)
        ]
        return torch.mean(torch.stack(logits_list), dim=0).unsqueeze(0)