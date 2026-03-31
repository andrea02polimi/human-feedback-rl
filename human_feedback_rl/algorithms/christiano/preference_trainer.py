import random
from typing import Dict

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
    """

    def __init__(
        self,
        preference_model: PreferenceModelFromReward,
        batch_size: int = 32,
        num_epochs: int = 10,
        logger: UnifiedLogger = None,
    ):
        self.preference_model = preference_model
        self.reward_model = preference_model.reward_model

        self.batch_size = batch_size
        self.logger = logger
        self.num_epochs = num_epochs
        self.global_epochs = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def train(self, dataset: PreferenceDataset):
        """Train on a preference dataset. Returns average training loss."""
        if len(dataset) == 0:
            return 0.0

        total_loss = 0.0
        total_updates = 0

        for epoch in range(self.num_epochs):
            # shuffle independently per ensemble
            n = len(dataset)
            bootstrap_indices_per_ensemble = [
                np.random.choice(n, size=n, replace=True).tolist()
                for _ in range(self.reward_model.n_ensembles)
            ]

            epoch_loss, n_steps = self._train_one_epoch(dataset, bootstrap_indices_per_ensemble)
            total_loss += epoch_loss
            total_updates += n_steps
            self.global_epochs += 1

            self.logger.record("loss", epoch_loss)
            self.logger.record("global_epochs", self.global_epochs)
            self.logger.dump()


    def evaluate(self, dataset: PreferenceDataset) -> float:
        """Evaluate reward model accuracy/loss on a preference dataset."""
        return RewardModelEvaluator(self.preference_model).evaluate(dataset)

    # -----------------------------------------------------------------------
    # Internal logic
    # -----------------------------------------------------------------------

    def _train_one_epoch(self, dataset, bootstrap_indices_per_ensemble) -> float:

        epoch_loss = 0.0
        n_steps = 0

        # iterate batches using SAME positions
        for batch_start in range(0, len(bootstrap_indices_per_ensemble[0]), self.batch_size):
            batch_indices_per_ensemble = [
                indices[batch_start: batch_start + self.batch_size]
                for indices in bootstrap_indices_per_ensemble
            ]

            epoch_loss += self._train_on_batch(
                dataset, batch_indices_per_ensemble
            )
            n_steps += 1

        return epoch_loss / max(n_steps, 1), n_steps
    

    def _train_on_batch(self, dataset, batch_indices_per_ensemble) -> float:
        total_batch_loss = 0.0

        for ensemble_idx, batch_indices in enumerate(batch_indices_per_ensemble):
            opt = self.reward_model.optimizers[ensemble_idx]
            opt.zero_grad()

            losses = []

            for idx in batch_indices:
                pair = dataset.pairs[idx]
                pref = dataset.preferences[idx]
                target = _preference_to_target(pref, self.reward_model.device)

                loss = self._compute_pair_loss(pair, target, ensemble_idx)
                losses.append(loss)

            # ONE backward per ensemble
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            opt.step()

            total_batch_loss += batch_loss.item()

        return total_batch_loss / len(batch_indices_per_ensemble)

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