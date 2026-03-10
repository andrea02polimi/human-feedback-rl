import torch


class RewardModel(torch.nn.Module):

    def __init__(self, obs_dim, n_actions):

        super().__init__()

        self.n_actions = n_actions

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + n_actions, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, state, action):

        action_onehot = torch.nn.functional.one_hot(
            action,
            num_classes=self.n_actions
        ).float()

        x = torch.cat([state, action_onehot], dim=1)

        return self.net(x)