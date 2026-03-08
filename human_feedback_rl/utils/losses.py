import torch
import torch.nn.functional as F


def imitation_loss(logits, expert_action):

    target = torch.tensor([expert_action], dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, target)


def preference_loss(logits, a_i, a_j, preferred, margin=0.2):

    q_i = logits[0, a_i]
    q_j = logits[0, a_j]

    if preferred == 0:
        diff = q_i - q_j
    else:
        diff = q_j - q_i

    return torch.relu(margin - diff)