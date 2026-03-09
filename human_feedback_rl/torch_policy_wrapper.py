import torch


class TorchPolicyWrapper:

    def __init__(self, policy):
        self.policy = policy

    def predict(self, obs):

        state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.policy(state)

        action = torch.argmax(logits, dim=1).item()

        return action