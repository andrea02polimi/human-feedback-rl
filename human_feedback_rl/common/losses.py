"""Bradley-Terry preference losses shared by training and evaluation."""

import torch as th

# Floor applied inside log() for numerical stability of the cross-entropy.
_LOG_EPS = 1e-7


def bradley_terry_probs(r1: th.Tensor, r2: th.Tensor) -> th.Tensor:
    """P(fragment 1 preferred) under the Bradley-Terry model.

    Args:
        r1, r2: per-pair fragment scores, shape (N,).
    Returns:
        Probabilities for (fragment 1, fragment 2), shape (N, 2).
    """
    prob1 = th.sigmoid(r1 - r2)
    return th.stack([prob1, 1 - prob1], dim=1)


def preference_nll(probs: th.Tensor, labels: th.Tensor) -> th.Tensor:
    """Soft cross-entropy between predicted probabilities and preference labels.

    Args:
        probs: Bradley-Terry probabilities, shape (N, 2).
        labels: preference labels summing to 1 per row, shape (N, 2).
    Returns:
        Scalar mean negative log-likelihood.
    """
    return -(labels * probs.clamp(min=_LOG_EPS).log()).sum(dim=1).mean()


def preference_accuracy(probs: th.Tensor, labels: th.Tensor) -> th.Tensor:
    """Fraction of pairs where the predicted winner matches the labelled winner."""
    return (probs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
