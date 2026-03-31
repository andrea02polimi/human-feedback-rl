"""
RewardTrainerHumLrn
===================
Trainer del reward model con loss combinata:

    L_total = L_pref + lambda_demo * L_demo

  - L_pref : Bradley-Terry cross-entropy su coppie di segmenti (identica a Christiano)
  - L_demo : margin ranking loss che impone R(demo) > R(agent_seg) + margin
             per ogni traiettoria dimostrativa vs un segmento agente casuale

Ogni membro dell'ensemble è addestrato su un bootstrap sample indipendente
del preference dataset, e vede le stesse dimostrazioni (non bootstrappate).
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from human_feedback_rl.common import (
    PreferenceDataset,
    PreferenceModelFromReward,
    UnifiedLogger,
)
from .demo_dataset import DemonstrationDataset
from human_feedback_rl.algorithms.christiano.preference_trainer import _preference_to_target


class RewardTrainerHumLrn:
    """
    Trainer reward model con loss ibrida preferenze + dimostrazioni.

    Args:
        preference_model : modello Bradley-Terry sopra EnsembleRewardModel
        lambda_demo      : peso della demonstration loss rispetto alla preference loss
        demo_margin      : margine nella ranking loss R(demo) > R(agent) + margin
    """

    def __init__(
        self,
        preference_model: PreferenceModelFromReward,
        batch_size: int = 32,
        num_epochs: int = 10,
        logger: UnifiedLogger = None,
        lambda_demo: float = 1.0,
        demo_margin: float = 1.0,
    ):
        self.preference_model = preference_model
        self.reward_model = preference_model.reward_model

        self.batch_size = batch_size
        self.logger = logger
        self.num_epochs = num_epochs
        self.lambda_demo = lambda_demo
        self.demo_margin = demo_margin
        self.global_epochs = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def train(
        self,
        pref_dataset: PreferenceDataset,
        demo_dataset: DemonstrationDataset | None,
    ) -> float:
        """
        Train reward model combinando preference loss e demonstration loss.
        Se demo_dataset è None, viene usata solo la preference loss.
        Ritorna la loss media totale.
        """
        if len(pref_dataset) == 0:
            return 0.0

        n = len(pref_dataset)
        total_loss = 0.0
        total_updates = 0

        for ensemble_idx in range(self.reward_model.n_ensembles):
            bootstrap_indices = np.random.choice(n, size=n, replace=True).tolist()

            for _ in range(self.num_epochs):
                epoch_loss, n_steps = self._train_one_epoch(
                    pref_dataset, demo_dataset, bootstrap_indices, ensemble_idx
                )
                total_loss += epoch_loss
                total_updates += n_steps
                self.global_epochs += 1

        return total_loss / max(total_updates, 1)

    def evaluate(self, pref_dataset: PreferenceDataset) -> float:
        """Evaluation sulla preference loss (validation set)."""
        # TODO: valutare separatamente anche la demonstration loss
        from human_feedback_rl.algorithms.christiano.preference_trainer import RewardModelEvaluator
        return RewardModelEvaluator(self.preference_model).evaluate(pref_dataset)

    # -----------------------------------------------------------------------
    # Epoch / batch logic
    # -----------------------------------------------------------------------

    def _train_one_epoch(
        self,
        pref_dataset: PreferenceDataset,
        demo_dataset: DemonstrationDataset,
        indices: list,
        ensemble_idx: int,
    ):
        shuffled = indices.copy()
        random.shuffle(shuffled)

        epoch_loss = 0.0
        n_steps = 0

        for batch_indices in self._iterate_minibatches(shuffled):
            loss = self._train_on_batch(pref_dataset, demo_dataset, batch_indices, ensemble_idx)
            epoch_loss += loss
            n_steps += 1

        return epoch_loss / max(n_steps, 1), n_steps

    def _iterate_minibatches(self, indices):
        for start in range(0, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size]

    def _train_on_batch(
        self,
        pref_dataset: PreferenceDataset,
        demo_dataset: DemonstrationDataset,
        batch_indices: list,
        ensemble_idx: int,
    ) -> float:
        opt = self.reward_model.optimizers[ensemble_idx]
        opt.zero_grad()

        # --- Preference loss ---
        pref_loss = sum(
            self._compute_pref_loss(
                pref_dataset.pairs[idx],
                _preference_to_target(pref_dataset.preferences[idx], self.reward_model.device),
                ensemble_idx,
            )
            for idx in batch_indices
        ) / len(batch_indices)

        # --- Demonstration loss (optional) ---
        if demo_dataset is not None and len(demo_dataset) > 0:
            demo_loss = self._compute_demo_loss(demo_dataset, ensemble_idx)
            loss = pref_loss + self.lambda_demo * demo_loss
        else:
            loss = pref_loss
        loss.backward()
        opt.step()

        return loss.item()

    # -----------------------------------------------------------------------
    # Loss computations
    # -----------------------------------------------------------------------

    def _compute_pref_loss(self, pair, target, ensemble_idx: int) -> torch.Tensor:
        """Cross-entropy Bradley-Terry loss per un membro dell'ensemble."""
        r1, r2 = self.preference_model.preference_logits_for_net(
            pair.seg1, pair.seg2, ensemble_idx
        )
        return F.cross_entropy(torch.stack([r1, r2]).unsqueeze(0), target)

    def _compute_demo_loss(
        self,
        demo_dataset: DemonstrationDataset,
        ensemble_idx: int,
    ) -> torch.Tensor:
        """
        Margin ranking loss: R(demo_seg) > R(agent_seg) + margin

        Per ogni traiettoria dimostrativa, campiona un segmento e confronta
        il suo predicted return con quello di un segmento agente casuale
        estratto dalla demo_dataset (usato come proxy di segmento sub-ottimale).

        TODO: usare segmenti dell'agente corrente (dal rollout buffer) invece
              di segmenti casuali dalla demo_dataset come termine negativo.
        TODO: implementare campionamento da DemonstrationDataset.sample()
        """
        raise NotImplementedError