import torch


def sample_two_actions(logits):

    probs = torch.softmax(logits, dim=1)

    actions = torch.multinomial(probs, 2, replacement=False)

    return actions[0,0].item(), actions[0,1].item()


def sample_topk_actions(logits, k=4):

    k = min(k, logits.shape[1])

    topk = torch.topk(logits, k).indices[0]

    pair = torch.randperm(k)[:2]

    return topk[pair[0]].item(), topk[pair[1]].item()