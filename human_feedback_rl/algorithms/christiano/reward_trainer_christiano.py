from human_feedback_rl.common import *
import torch
import torch.nn.functional as F
import random
from typing import Dict


# ---------------------------------------------------------------------------
# Reward Trainer (Christiano et al. 2017)
# ---------------------------------------------------------------------------

class RewardTrainerChristiano:
    """
    Trainer for an ensemble reward model using preference comparisons.

    Each ensemble member is trained independently using
    cross-entropy loss on pairwise preferences.
    """

    def __init__(
        self,
        preference_model,
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

    def train(self, dataset: PreferenceDataset) -> float:
        """
        Train the reward model on a preference dataset.

        Args:
            dataset: dataset of segment pairs + preferences
            n_epochs: optional override

        Returns:
            average training loss
        """
        if len(dataset) == 0:
            return 0.0

        total_loss = 0.0
        total_updates = 0

        for epoch in range(self.num_epochs):
            epoch_loss, n_steps = self._train_one_epoch(dataset)

            total_loss += epoch_loss
            total_updates += n_steps
            self.global_epochs += 1

            if self.logger:
                self.logger.record("reward_model/train/epoch_loss", epoch_loss)
                self.logger.record("reward_model/train/epoch", epoch)
                self.logger.record("timescales/global_reward_trainer_epochs", self.global_epochs)
                self.logger.dump(step=self.global_epochs)

        return total_loss / max(total_updates, 1)


    def evaluate(self, dataset: PreferenceDataset) -> Dict[str, float]:
        """
        Thin wrapper for evaluation.
        """
        evaluator = RewardModelEvaluator(self.preference_model)
        return evaluator.evaluate(dataset)


    # -----------------------------------------------------------------------
    # Internal logic
    # -----------------------------------------------------------------------

    def _train_one_epoch(self, dataset: PreferenceDataset):
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        epoch_loss = 0.0
        n_steps = 0

        for batch_indices in self._iterate_minibatches(indices):
            batch_loss = self._train_on_batch(dataset, batch_indices)

            epoch_loss += batch_loss
            n_steps += 1

        return epoch_loss / max(n_steps, 1), n_steps

    def _iterate_minibatches(self, indices):
        for start in range(0, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size]

    def _train_on_batch(self, dataset: PreferenceDataset, batch_indices):
        self._zero_grad_all()

        total_batch_loss = 0.0

        for idx in batch_indices:
            pair = dataset.pairs[idx]
            preference = dataset.targets[idx]

            target = self._preference_to_target(preference)

            loss = self._compute_pair_loss(pair, target)
            loss.backward()

            total_batch_loss += loss.item()

        self._step_all_optimizers()

        return total_batch_loss / len(batch_indices)

    # -----------------------------------------------------------------------
    # Core computations
    # -----------------------------------------------------------------------

    def _compute_pair_loss(self, pair, target):
        """
        Compute loss across all ensemble members for one pair.
        """
        loss_sum = 0.0

        for ensemble_id in range(self.reward_model.n_ensembles):
            r1, r2 = self.preference_model.preference_logits_for_net(
                pair.seg1, pair.seg2, ensemble_id
            )

            logits = torch.stack([r1, r2]).unsqueeze(0)
            loss = F.cross_entropy(logits, target)

            loss_sum += loss

        return loss_sum / self.reward_model.n_ensembles

    def _preference_to_target(self, preference):
        """
        Convert preference label to class index.
        """
        target_idx = preference.label.index(max(preference.label))

        return torch.tensor(
            [target_idx],
            dtype=torch.long,
            device=self.reward_model.device,
        )

    # -----------------------------------------------------------------------
    # Optimizer utilities
    # -----------------------------------------------------------------------

    def _zero_grad_all(self):
        for opt in self.reward_model.optimizers:
            opt.zero_grad()

    def _step_all_optimizers(self):
        for opt in self.reward_model.optimizers:
            opt.step()


# ---------------------------------------------------------------------------
# Evaluation Module
# ---------------------------------------------------------------------------

class RewardModelEvaluator:
    """
    Evaluate a reward model on a preference dataset.
    Computes accuracy and average loss.
    """

    def __init__(self, preference_model: PreferenceModelFromReward):
        self.preference_model = preference_model
        self.reward_model = preference_model.reward_model

    def evaluate(self, dataset: PreferenceDataset) -> Dict[str, float]:
        if len(dataset) == 0:
            return {"accuracy": 0.0, "loss": 0.0}

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(len(dataset)):
                pair = dataset.pairs[i]
                preference = dataset.targets[i]

                target = self._preference_to_target(preference)

                logits = self._ensemble_logits(pair)

                loss = F.cross_entropy(logits, target)
                total_loss += loss.item()

                pred = torch.argmax(logits, dim=1)
                correct += (pred == target).sum().item()
                total += 1

        return total_loss / total

    def _ensemble_logits(self, pair):
        """
        Average logits across ensemble members.
        """
        logits_list = []

        for k in range(self.reward_model.n_ensembles):
            r1, r2 = self.preference_model.preference_logits_for_net(
                pair.seg1, pair.seg2, k
            )
            logits_list.append(torch.stack([r1, r2]))

        mean_logits = torch.mean(torch.stack(logits_list), dim=0)
        return mean_logits.unsqueeze(0)

    def _preference_to_target(self, preference):
        idx = preference.label.index(max(preference.label))
        return torch.tensor(
            [idx],
            dtype=torch.long,
            device=self.reward_model.device,
        )