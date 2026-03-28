import torch
from stable_baselines3.common.policies import ActorCriticPolicy


class BCPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
            net_arch=dict(pi=[64, 64], vf=[])
        )

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, env, **kwargs):
        policy = cls(
            env.observation_space,
            env.action_space,
            lr_schedule=lambda _: 1e-3,
            **kwargs,
        )
        policy.load_state_dict(torch.load(path, weights_only=True))
        return policy