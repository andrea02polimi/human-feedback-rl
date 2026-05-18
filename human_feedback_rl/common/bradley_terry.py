import torch as th
import torch.nn.functional as F


def BradleyTerry(r1: th.Tensor, r2: th.Tensor) -> th.Tensor:
    """Bradley-Terry preference model: P(1 > 2) = exp(r1) / (exp(r1) + exp(r2)).

    Args: r1, r2 of shape (batch,). Returns (batch, 2) preference probs.
    """
    logits = th.stack([r1, r2], dim=1)
    return F.softmax(logits, dim=1)

