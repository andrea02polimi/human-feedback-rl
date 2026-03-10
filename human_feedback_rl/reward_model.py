import torch


class RewardModel(torch.nn.Module):

    def __init__(self, obs_dim, n_actions):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + n_actions, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        self.n_actions = n_actions

    def forward(self, state, action):

        state = torch.as_tensor(state, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        action = torch.as_tensor(action, dtype=torch.long)

        if action.dim() == 0:
            action = action.unsqueeze(0)

        one_hot = torch.nn.functional.one_hot(
            action,
            num_classes=self.n_actions
        ).float()

        x = torch.cat([state, one_hot.view(state.shape[0], -1)], dim=-1)

        return self.net(x).squeeze(-1)

    def update_reward_model(self, seg1, seg2, preference, optimizer):

        r1 = torch.stack([
            self.forward(step.state, step.action)
            for step in seg1
        ]).mean()

        r2 = torch.stack([
            self.forward(step.state, step.action)
            for step in seg2
        ]).mean()

        diff = torch.clamp(r1 - r2, -10, 10)

        target = torch.tensor(
            1.0 if preference == 0 else 0.0,
            dtype=diff.dtype
        )

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            diff.view(1),
            target.view(1)
        )

        loss = loss * 0.5

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        optimizer.step()

        return loss.item()