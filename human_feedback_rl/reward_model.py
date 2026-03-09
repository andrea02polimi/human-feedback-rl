import torch


class RewardModel(torch.nn.Module):

    def __init__(self, obs_dim, n_actions):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + n_actions, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1)
        )

        self.n_actions = n_actions

    def forward(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        action_tensor = torch.as_tensor(action, dtype=torch.long, device=state.device)

        if action_tensor.dim() == 0:
            action_tensor = action_tensor.unsqueeze(0)

        one_hot = torch.nn.functional.one_hot(
            action_tensor,
            num_classes=self.n_actions
        ).float()

        x = torch.cat([state, one_hot.view(state.shape[0], -1)], dim=-1)

        return self.net(x).squeeze(-1)

    def update_reward_model(self, seg1, seg2, preference, optimizer):
        r1 = torch.stack([
            self.forward(step.state, step.action)
            for step in seg1
        ]).sum()

        r2 = torch.stack([
            self.forward(step.state, step.action)
            for step in seg2
        ]).sum()

        diff = torch.clamp(r1 - r2, -10, 10)

        target = torch.tensor(
            1.0 if preference == 0 else 0.0,
            dtype=diff.dtype,
            device=diff.device
        )

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            diff.view(1),
            target.view(1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()