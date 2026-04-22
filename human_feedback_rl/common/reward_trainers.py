import abc
import random
from collections.abc import Iterator
from typing import Mapping, NamedTuple, Sequence

import torch as th
import torch.nn as nn

from .reward_nets import RewardEnsemble
from .types import FragmentPair, Preference
from .preference_models import PreferenceModelFromReward

# Incremental implementation note:
# 1. done: loss container and abstract loss interface
# 2. done: cross-entropy reward loss with math comments
# 3. done: abstract reward trainer
# 4. done: Preference -> scalar target μ helper
# Next planned snippets:
# - mini-batch collation for (σ, μ)
# - basic trainer __init__
# - one training epoch
# - full train loop
# - ensemble trainer


class LossAndMetrics(NamedTuple):
    """Output of a reward-loss computation.

    `loss` is the scalar optimization objective used for backpropagation.
    `metrics` contains detached quantities such as accuracy for logging only.
    """

    loss: th.Tensor
    metrics: Mapping[str, th.Tensor]


class RewardLoss(nn.Module, abc.ABC):
    """Abstract objective over preference comparisons.

    Given fragment-pair inputs `σ` and target preferences `μ`,
    a concrete subclass computes a scalar objective `L(φ)` for the
    current preference model parameters `φ`.
    """

    @abc.abstractmethod
    def forward(
        self,
        fragment_pairs: Sequence[FragmentPair],
        preferences: Sequence[Preference],
        preference_model: PreferenceModelFromReward,
    ) -> LossAndMetrics:
        """Compute the loss and auxiliary metrics."""


class CrossEntropyRewardLoss(RewardLoss):
    """Cross-entropy on pairwise preference distributions.

    For each fragment pair `σ = (σ_1, σ_2)`, the preference model predicts

        p_φ = P_φ(σ_1 is preferred to σ_2)

    and the dataset provides a soft target distribution

    where typically:
      - (1.0, 0.0) means σ_1 is preferred
      - (0.0, 1.0) means σ_2 is preferred
      - (0.5, 0.5) means tie / indifference

    The per-example loss is the two-class cross-entropy

        L = -(μ_1 log p_φ + μ_2 log(1 - p_φ)).
    """

    def forward(
        self,
        fragment_pairs: Sequence[FragmentPair],
        preferences: Sequence[Preference],
        preference_model: PreferenceModelFromReward,
    ) -> LossAndMetrics:
        # p_φ = P_φ(σ_1 > σ_2), clamped away from 0 and 1 so the
        # logarithms inside cross-entropy remain numerically stable.
        probs = preference_model(fragment_pairs).clamp(1e-7, 1.0 - 1e-7)
        pair_probs = th.stack([probs, 1.0 - probs], dim=1)
        preferences_th = th.as_tensor(
            [[preference.pref1, preference.pref2] for preference in preferences],
            dtype=th.float32,
            device=probs.device,
        )

        # Hard prediction from the two-class distribution.
        predictions = pair_probs.argmax(dim=1)
        # Hard target: argmax([μ_1, μ_2]). Ties default to class 0.
        targets = preferences_th.argmax(dim=1)

        metrics = {
            # Mean classification accuracy over the batch, detached because it is
            # only a logging metric and not part of the gradient computation.
            "accuracy": (predictions == targets).float().mean().detach().cpu(),
        }
        # Batch-mean cross-entropy with soft two-class targets:
        #   L = mean(-sum_i μ_i log p_i).
        loss = -(preferences_th * pair_probs.log()).sum(dim=1).mean()
        return LossAndMetrics(loss=loss, metrics=metrics)


class RewardTrainer(abc.ABC):
    """Base trainer for fitting a preference model on labeled pairs `(σ, μ)`."""

    def __init__(
        self,
        preference_model: PreferenceModelFromReward,
        logger=None,
    ) -> None:
        self.preference_model = preference_model
        self.logger = logger

    def train(self, dataset, epoch_multiplier: float = 1.0):
        """Train on a dataset of fragment-pair / target-preference samples `(σ, μ)`."""
        return self._train(dataset, epoch_multiplier=epoch_multiplier)

    @abc.abstractmethod
    def _train(self, dataset, epoch_multiplier: float = 1.0):
        """Run the concrete training procedure."""


def collate_preference_batch(
    batch: Sequence[tuple[FragmentPair, Preference]],
) -> tuple[list[FragmentPair], list[Preference]]:
    """Convert a `(FragmentPair, Preference)` minibatch into aligned sequences."""
    fragment_pairs = [pair for pair, _ in batch]
    preferences = [preference for _, preference in batch]
    return fragment_pairs, preferences


class BasicRewardTrainer(RewardTrainer):
    """Mini-batch trainer for a single preference model."""

    def __init__(
        self,
        preference_model: PreferenceModelFromReward,
        loss: RewardLoss,
        optimizer: th.optim.Optimizer,
        batch_size: int = 32,
        epochs: int = 1,
        logger=None,
    ) -> None:
        super().__init__(preference_model=preference_model, logger=logger)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")

        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

    def _iterate_minibatches(
        self,
        dataset: Sequence[tuple[FragmentPair, Preference]],
    ) -> Iterator[Sequence[tuple[FragmentPair, Preference]]]:
        for start in range(0, len(dataset), self.batch_size):
            yield dataset[start : start + self.batch_size]

    def _train_one_batch(
        self,
        batch: Sequence[tuple[FragmentPair, Preference]],
    ) -> LossAndMetrics:
        fragment_pairs, preferences = collate_preference_batch(batch)
        loss_and_metrics = self.loss(
            fragment_pairs=fragment_pairs,
            preferences=preferences,
            preference_model=self.preference_model,
        )
        self.optimizer.zero_grad()
        loss_and_metrics.loss.backward()
        self.optimizer.step()
        return loss_and_metrics

    def _train(
        self,
        dataset: Sequence[tuple[FragmentPair, Preference]],
        epoch_multiplier: float = 1.0,
    ) -> Mapping[str, float]:
        if not dataset:
            return {"loss": 0.0, "accuracy": 0.0}

        total_epochs = max(1, int(round(self.epochs * epoch_multiplier)))
        loss_sum = 0.0
        accuracy_sum = 0.0
        num_batches = 0

        for _ in range(total_epochs):
            epoch_dataset = list(dataset)
            random.shuffle(epoch_dataset)

            for batch in self._iterate_minibatches(epoch_dataset):
                loss_and_metrics = self._train_one_batch(batch)
                loss_sum += float(loss_and_metrics.loss.detach().cpu())
                accuracy = loss_and_metrics.metrics.get("accuracy")
                if accuracy is not None:
                    accuracy_sum += float(accuracy)
                num_batches += 1

        result = {
            "loss": loss_sum / num_batches,
            "accuracy": accuracy_sum / num_batches,
        }

        if self.logger is not None:
            for key, value in result.items():
                self.logger.record(key, value)

        return result


class EnsembleRewardTrainer(RewardTrainer):
    """Bootstrap training for each member of a reward ensemble."""

    def __init__(
        self,
        reward_model: RewardEnsemble,
        optimizer_cls: type[th.optim.Optimizer] = th.optim.Adam,
        loss_factory=None,
        optimizer_kwargs: Mapping[str, object] | None = None,
        batch_size: int = 32,
        epochs: int = 1,
        logger=None,
    ) -> None:
        super().__init__(preference_model=PreferenceModelFromReward(reward_model), logger=logger)
        self.reward_model = reward_model
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.loss_factory = loss_factory or CrossEntropyRewardLoss

        self.member_trainers = [
            BasicRewardTrainer(
                preference_model=PreferenceModelFromReward(member),
                loss=self.loss_factory(),
                optimizer=optimizer_cls(member.parameters(), **self.optimizer_kwargs),
                batch_size=batch_size,
                epochs=epochs,
                logger=None,
            )
            for member in reward_model.members
        ]

    def _bootstrap_dataset(
        self,
        dataset: Sequence[tuple[FragmentPair, Preference]],
    ) -> list[tuple[FragmentPair, Preference]]:
        return [random.choice(dataset) for _ in range(len(dataset))]

    def _train(
        self,
        dataset: Sequence[tuple[FragmentPair, Preference]],
        epoch_multiplier: float = 1.0,
    ) -> Mapping[str, float]:
        if not dataset:
            return {"loss": 0.0, "accuracy": 0.0}

        loss_sum = 0.0
        accuracy_sum = 0.0

        for trainer in self.member_trainers:
            bootstrap_dataset = self._bootstrap_dataset(dataset)
            result = trainer.train(bootstrap_dataset, epoch_multiplier=epoch_multiplier)
            loss_sum += result["loss"]
            accuracy_sum += result["accuracy"]

        result = {
            "loss": loss_sum / len(self.member_trainers),
            "accuracy": accuracy_sum / len(self.member_trainers),
        }

        if self.logger is not None:
            for key, value in result.items():
                self.logger.record(key, value)

        return result
