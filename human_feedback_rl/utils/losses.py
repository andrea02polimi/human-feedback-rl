import torch
import torch.nn.functional as F


def imitation_loss(logits, expert_action):

    target = torch.tensor([expert_action], dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, target)


def preference_loss(r1, r2, probs):

    logits = torch.stack([r1, r2], dim=1)

    target = torch.tensor(probs)

    pred = torch.log_softmax(logits, dim=1)

    loss = -(target * pred).sum(dim=1).mean()

    return loss