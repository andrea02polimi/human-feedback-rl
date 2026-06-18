import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer

from .env_wrappers import ego_status_to_onehot

_STATUS_DIM = 7


class RelabelReplayBuffer(ReplayBuffer):
    """SB3 ReplayBuffer that also stores each transition's next ego-status one-hot,
    so the stored rewards can be recomputed ("relabelled") with an updated reward
    model.

    The reward model changes during training, so an off-policy agent like SAC keeps
    reusing transitions whose rewards were predicted by a stale model. SumoRewardNet
    needs ``next_status`` (a 7-dim one-hot) to predict a reward, and the standard
    ReplayBuffer does not keep it -- hence this subclass. Call ``relabel()`` after
    each reward-model update to overwrite ``self.rewards``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mirrors self.rewards layout: (buffer_size, n_envs, status_dim)
        self.next_status = np.zeros(
            (self.buffer_size, self.n_envs, _STATUS_DIM), dtype=np.float32
        )

    def add(self, obs, next_obs, action, reward, done, infos):
        # Capture the status BEFORE super().add() advances self.pos.
        for i, info in enumerate(infos):
            self.next_status[self.pos, i] = ego_status_to_onehot(
                info.get("ego_status", "running")
            )
        super().add(obs, next_obs, action, reward, done, infos)

    @th.no_grad()
    def relabel(self, reward_model, reward_mean: float = 0.0, reward_std: float = 1.0) -> None:
        """Recompute the reward of every stored transition with ``reward_model``.

        Mirrors EnvRewardWrapper's status-based branch: reward =
        reward_model.predict(obs, action, next_status, done), then the running
        standardization (``reward_mean``/``reward_std``) so relabelled rewards stay
        on the same scale as freshly collected ones.
        """
        upper = self.buffer_size if self.full else self.pos
        if upper == 0:
            return

        obs         = self.observations[:upper].reshape(-1, *self.obs_shape)
        actions     = self.actions[:upper].reshape(-1, self.action_dim)
        next_status = self.next_status[:upper].reshape(-1, _STATUS_DIM)
        # self.dones is the combined done (terminated | truncated), matching the raw
        # done that EnvRewardWrapper feeds SumoRewardNet in its status branch.
        dones       = self.dones[:upper].reshape(-1).astype(np.float32)

        rew = reward_model.predict(obs, actions, next_status, dones)
        rew = (rew - reward_mean) / (reward_std + 1e-8)
        self.rewards[:upper] = rew.reshape(upper, self.n_envs)
