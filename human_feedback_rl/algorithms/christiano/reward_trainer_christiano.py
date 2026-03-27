from common import *
import torch
import torch.nn.functional as F
import random

# ---------------------------------------------------------------------------
# Reward trainer
# ---------------------------------------------------------------------------

class RewardTrainerChristiano:
    """
    Trains EnsembleRewardModel on a PreferenceDataset.

    Loss: mean cross-entropy preference loss (Christiano et al. eq. 1)
    applied independently to each ensemble member.
    """

    def __init__(
        self,
        preference_model: PreferenceModelFromReward,
        batch_size: int = 32,
        n_epochs: int = 10,
    ):
        self.preference_model = preference_model
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def train(self, dataset: PreferenceDataset) -> float:
        """
        Train on the preference dataset.

        Returns:
            mean loss over the training run.
        """
        if len(dataset) == 0:
            return 0.0

        rm = self.preference_model.reward_model
        total_loss = 0.0
        n_steps = 0

        for _ in range(self.n_epochs):
            indices = list(range(len(dataset)))
            random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                for opt in rm.optimizers:
                    opt.zero_grad()

                batch_loss = 0.0
                for i in batch_idx:
                    pair = dataset.pairs[i]
                    pref = dataset.targets[i]
                    # label (1,0) -> target=0 (seg1 preferred)
                    # label (0,1) -> target=1 (seg2 preferred)
                    target = torch.tensor(
                        [pref.label.index(max(pref.label))],
                        dtype=torch.long,
                        device=rm.device,
                    )

                    for k in range(rm.n_ensembles):
                        r1, r2 = self.preference_model.preference_logits_for_net(
                            pair.seg1, pair.seg2, k
                        )
                        logits = torch.stack([r1, r2]).unsqueeze(0)
                        loss = F.cross_entropy(logits, target)
                        loss.backward()
                        batch_loss += loss.item()

                for opt in rm.optimizers:
                    opt.step()

                total_loss += batch_loss / len(batch_idx)
                n_steps += 1

        return total_loss / max(n_steps, 1)
