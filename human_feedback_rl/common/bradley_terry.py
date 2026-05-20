import torch as th
import torch.nn.functional as F


# def BradleyTerry(r1: th.Tensor, r2: th.Tensor, temperature: float = 1.0) -> th.Tensor:
#     """Bradley-Terry preference model: P(1 > 2) = exp(r1/T) / (exp(r1/T) + exp(r2/T)).
#
#     Args: r1, r2 of shape (batch,). Returns (batch, 2) preference probs.
#     temperature: scales logits before softmax; higher values soften the
#         distribution and prevent gradient saturation when reward gaps are large.
#     """
#     temp = th.tensor(temperature)
#     logits = th.stack([r1, r2], dim=1) / temp
#     return F.softmax(logits, dim=1)



def BradleyTerry(
    r1: th.Tensor,
    r2: th.Tensor,
    temperature: float = 1.0,
) -> th.Tensor:
    """Bradley-Terry preference model using sigmoid.
    Args:
        r1, r2: tensors of shape (batch,)
        temperature: softening factor
    Returns:
        Tensor of shape (batch, 2):
        [P(1 > 2), P(2 > 1)]
    """
    temp = th.tensor(temperature, device=r1.device, dtype=r1.dtype)
    p1 = th.sigmoid((r1 - r2) / temp)
    p2 = 1.0 - p1
    return th.stack([p1, p2], dim=1)

