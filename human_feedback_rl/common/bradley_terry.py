import torch as th
import torch.nn.functional as F


def BradleyTerry(r1: th.Tensor, r2: th.Tensor, temperature: float = 1.0) -> th.Tensor:
    """Bradley-Terry preference model: P(1 > 2) = exp(r1/T) / (exp(r1/T) + exp(r2/T)).

    Args: r1, r2 of shape (batch,). Returns (batch, 2) preference probs.
    temperature: scales logits before softmax; higher values soften the
        distribution and prevent gradient saturation when reward gaps are large.
    """
    logits = th.stack([r1, r2], dim=1) / temperature
    return F.softmax(logits, dim=1)

